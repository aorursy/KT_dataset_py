import pandas as pd

import numpy as np

import os

import warnings



from lightgbm import LGBMRegressor

from sklearn.base import BaseEstimator, RegressorMixin, clone

from sklearn.metrics import mean_squared_log_error

from IPython.display import HTML



pd.set_option("max_columns", 500)





def input_file(file):

    path = f"../input/ashrae-energy-prediction/{file}"

    if not os.path.exists(path): return path + ".gz"

    return path



def compress_dataframe(df):

    result = df.copy()

    for col in result.columns:

        col_data = result[col]

        dn = col_data.dtype.name

        if dn == "object":

            result[col] = pd.to_numeric(col_data.astype("category").cat.codes, downcast="integer")

        elif dn == "bool":

            result[col] = col_data.astype("int8")

        elif dn.startswith("int") or (col_data.round() == col_data).all():

            result[col] = pd.to_numeric(col_data, downcast="integer")

        else:

            result[col] = pd.to_numeric(col_data, downcast='float')

    return result



def read_train():

    df = pd.read_csv(input_file("train.csv"), parse_dates=["timestamp"])

    df.timestamp = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600

    return compress_dataframe(df)



def read_building_metadata():

    return compress_dataframe(pd.read_csv(

        input_file("building_metadata.csv")).fillna(-1)).set_index("building_id")



site_GMT_offsets = [-5, 0, -7, -5, -8, 0, -5, -5, -5, -6, -7, -5, 0, -6, -5, -5]



def read_weather_train(fix_timestamps=True, interpolate_na=True, add_na_indicators=True):

    df = pd.read_csv(input_file("weather_train.csv"), parse_dates=["timestamp"])

    df.timestamp = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600

    if fix_timestamps:

        GMT_offset_map = {site: offset for site, offset in enumerate(site_GMT_offsets)}

        df.timestamp = df.timestamp + df.site_id.map(GMT_offset_map)

    if interpolate_na:

        site_dfs = []

        for site_id in df.site_id.unique():

            # Make sure that we include all possible hours so that we can interpolate evenly

            site_df = df[df.site_id == site_id].set_index("timestamp").reindex(range(8784))

            site_df.site_id = site_id

            for col in [c for c in site_df.columns if c != "site_id"]:

                if add_na_indicators: site_df[f"had_{col}"] = ~site_df[col].isna()

                site_df[col] = site_df[col].interpolate(limit_direction='both', method='linear')

                # Some sites are completely missing some columns, so use this fallback

                site_df[col] = site_df[col].fillna(df[col].median())

            site_dfs.append(site_df)

        df = pd.concat(site_dfs).reset_index()  # make timestamp back into a regular column

    elif add_na_indicators:

        for col in df.columns:

            if df[col].isna().any(): df[f"had_{col}"] = ~df[col].isna()

    return compress_dataframe(df).set_index(["site_id", "timestamp"])



def combined_train_data(fix_timestamps=True, interpolate_na=True, add_na_indicators=True):

    Xy = compress_dataframe(read_train().join(read_building_metadata(), on="building_id").join(

        read_weather_train(fix_timestamps, interpolate_na, add_na_indicators),

        on=["site_id", "timestamp"]).fillna(-1))

    return Xy.drop(columns=["meter_reading"]), Xy.meter_reading



def _drop_electrical_zeros(X, y):

    X = X[(y > 0) | (X.meter != 0)]

    y = y.reindex(X.index)

    return X, y



def _drop_missing_site_0(X, y):

    X = X[(X.timestamp >= 3378) | (X.site_id != 0) | (X.meter != 0)]

    y = y.reindex(X.index)

    return X, y



def _add_time_features(X):

    return X.assign(tm_day_of_week=((X.timestamp // 24) % 7), tm_hour_of_day=(X.timestamp % 24))



def np_sample(a, frac):

    return a if frac == 1 else np.random.choice(a, int(len(a) * frac), replace=False)



def make_8121_splits(X, sample_frac):

    np.random.seed(0)

    time_sorted_idx = np.argsort(X.timestamp.values, kind='stable')

    sections = np.array_split(time_sorted_idx, 12)

    folds = []

    for start_ix in range(0, 12, 2):

        val_idxs = np.concatenate(sections[start_ix:start_ix + 2])  # no modulo necessary

        train_idxs = np.concatenate(

            [sections[ix % 12] for ix in range(start_ix + 3, start_ix + 11)])

        folds.append((np_sample(train_idxs, sample_frac), np_sample(val_idxs, sample_frac)))

    return folds



def make_cv_predictions(model, split, X, y, drop_electrical_zeros, verbose=True):

    preds = []

    for ix, (train_fold, val_fold) in enumerate(split):

        Xt = X.iloc[train_fold]

        yt = y.reindex_like(Xt)

        if drop_electrical_zeros:

            Xt, yt = _drop_electrical_zeros(Xt, yt)

        Xv = X.iloc[val_fold]

        yv = y.reindex_like(Xv)

        if verbose: print(f"Testing split {ix}: {len(Xt)} train rows & {len(Xv)} val rows")

        model.fit(Xt, yt)

        preds.append(pd.DataFrame(dict(target=yv, prediction=model.predict(Xv)), index=yv.index))

    result = pd.concat(preds).sort_index()

    return result.target, result.prediction



categorical_columns = [

    "building_id", "meter", "site_id", "primary_use", "had_air_temperature", "had_cloud_coverage",

    "had_dew_temperature", "had_precip_depth_1_hr", "had_sea_level_pressure", "had_wind_direction",

    "had_wind_speed", "tm_day_of_week", "tm_hour_of_day"

]



class LGBMWrapper(BaseEstimator, RegressorMixin):

    def __init__(self, categorical_feature=None, **params):

        self.model = LGBMRegressor(**params)

        self.categorical_feature = categorical_feature



    def fit(self, X, y):

        with warnings.catch_warnings():

            cats = None if self.categorical_feature is None else list(

                X.columns.intersection(self.categorical_feature))

            warnings.filterwarnings("ignore",

                                    "categorical_feature in Dataset is overridden".lower())

            self.model.fit(X, y, **({} if cats is None else {"categorical_feature": cats}))

            self.feature_importances_ = self.model.feature_importances_

            return self



    def predict(self, X):

        return self.model.predict(X)



    def get_params(self, deep=True):

        return {**self.model.get_params(deep), "categorical_feature": self.categorical_feature}



    def set_params(self, **params):

        ctf = params.pop("categorical_feature", None)

        if ctf is not None: self.categorical_feature = ctf

        self.model.set_params(params)
def run_experiment(n_estimators, sample_frac=1):

    X, y = combined_train_data()



    # Reduce evaluation cost by subsampling the data

    X = X.sample(frac=sample_frac).sort_index()

    y = y.reindex(X.index)



    # Additional preprocessing

    X, y = _drop_missing_site_0(X, y)

    X = compress_dataframe(_add_time_features(X))

    y = np.log1p(y)



    model = LGBMWrapper(random_state=0, n_jobs=-1, n_estimators=n_estimators,

                        categorical_feature=categorical_columns)

    contribution_chart = pd.DataFrame(columns=["building_id", "meter", "RMSLE", "contribution"])



    for building in sorted(X.building_id.unique()):

        for meter in range(4):

            X_subset = X[(X.building_id == building) & (X.meter == meter)]

            y_subset = y.reindex(X_subset.index)

            if len(y_subset) == 0: continue

            splits = make_8121_splits(X_subset, 1)  # We already subsampled, so no need to resample

            X_subset = X_subset.drop(columns="timestamp")



            cv_y, cv_prediction = make_cv_predictions(model, splits, X_subset, y_subset,

                                                      drop_electrical_zeros=False, verbose=False)

            sle = np.square(cv_y - cv_prediction).sum()

            rmsle = np.sqrt(sle / len(cv_y))

            contribution = np.sqrt(sle / len(y))

            contribution_chart.loc[len(contribution_chart)] = (building, meter, rmsle, contribution)

    return contribution_chart
def fractions_report(chart):

    contribution = chart.contribution

    cum_sum = (contribution.sort_values(ascending=False) / contribution.sum()).cumsum()

    for frac in [0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 1]:

        count = cum_sum[cum_sum < frac].count()+1  # Add one to account for boundary issues

        frac_frac = count / len(cum_sum)

        print(f"{count: 5} meters ({frac_frac:6.1%} of total) account for {frac:4.0%} of error.")
%matplotlib inline

import matplotlib.pyplot as plt



def plot_contributions(chart):

    for meter in range(4):

        subchart = chart[chart.meter == meter].sort_values("building_id")

        ax = plt.figure(0, (24, 8)).add_subplot(111)

        ax.set(xlim=(0, chart.building_id.max() + 1), ylim=(0, chart.contribution.max() * 1.05))

        ax.bar(subchart.building_id, subchart.contribution, width=1, label=f"meter {meter}")

        ax.legend()

        plt.show()
contribution_chart = run_experiment(10, 1)

contribution_chart.sort_values("contribution", ascending=False).to_csv("contribution_chart.csv", index=False)
fractions_report(contribution_chart)
display(contribution_chart.groupby("meter").contribution.sum().to_frame().T)
display(contribution_chart.sort_values("contribution", ascending=False).reset_index(drop=True).head(25))
plot_contributions(contribution_chart)