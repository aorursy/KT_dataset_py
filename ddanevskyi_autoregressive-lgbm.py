import os
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline
import lightgbm as lgb
from tqdm.notebook import tqdm
from sklearn.preprocessing import LabelEncoder
train_df = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

test_df = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

submission_df = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")
region_metadata = pd.read_csv("../input/covid19-forecasting-metadata/region_metadata.csv")
def transform_geo_location(df):

    lat = df.lat.values

    lon = df.lon.values

    

    df["lat_sin"] = np.sin(2 * np.pi * lat / 360)

    df["lat_cos"] = np.cos(2 * np.pi * lat / 360)

    df["lon_sin"] = np.sin(2 * np.pi * lon / 360)

    df["lon_cos"] = np.cos(2 * np.pi * lon / 360)

    

    return df
region_metadata = transform_geo_location(region_metadata)

geo_location_columns = ["lat_sin", "lat_cos", "lon_sin", "lon_cos"]
country_data = pd.read_csv("../input/countries-of-the-world/countries of the world.csv")

country_data.Country = country_data.Country.apply(lambda c: c.rstrip(" "))

country_data.rename(

    columns={

        "Country": "Country_Region",

    },

    inplace=True

)
country_meta_columns = [

    'Coastline (coast/area ratio)',

    'Net migration',

    'Infant mortality (per 1000 births)',

    'GDP ($ per capita)',

    'Literacy (%)',

    'Phones (per 1000)',

    'Arable (%)',

    'Crops (%)',

    'Other (%)',

    'Climate',

    'Birthrate',

    'Deathrate',

    'Agriculture',

    'Industry',

    'Service'

]
for col in country_meta_columns:

    # country_data[col].fillna(-1e6, inplace=True)

    country_data[col] = (

        country_data[col]

        .fillna(-1000)

        .apply(lambda v: v.replace(",", ".") if isinstance(v, str) else v)

        .astype(np.float32)

    )
lockdown_meta_df = pd.read_csv("../input/covid19-lockdown-dates-by-country/countryLockdowndates.csv")

lockdown_meta_df.rename(

    columns={

        "Country/Region": "Country_Region",

        "Province": "Province_State",

        "Date": "LockdownDate",

            "Type": "LockdownType"

    },

    inplace=True

)

lockdown_meta_df.drop("Reference", axis=1, inplace=True)



def convert_lockdown_date(d):

    if isinstance(d, str):

        return "-".join(reversed(d.split("/")))

    else:

        return "2020-12-31"



lockdown_meta_df.LockdownDate = lockdown_meta_df.LockdownDate.apply(convert_lockdown_date)
train_df = train_df.merge(lockdown_meta_df, on=["Province_State", "Country_Region"], how="left")

train_df = train_df.merge(country_data, on=["Country_Region"], how="left")
train_df["days_since_lockdown"] = np.clip(

    (pd.to_datetime(train_df.Date) - pd.to_datetime(train_df.LockdownDate)).dt.days,

    a_min=-1,

    a_max=None

)

train_df["lockdown_type"] = [

    t if d >= 0 else "None" for t, d in zip (train_df.LockdownType, train_df.days_since_lockdown)

]

lockdown_type_encoder = LabelEncoder().fit(train_df.lockdown_type)

train_df["lockdown_type"] = lockdown_type_encoder.transform(train_df.lockdown_type)
train_df["ConfirmedCases"] = np.log1p(train_df.ConfirmedCases)

train_df["Fatalities"] = np.log1p(train_df.Fatalities)
def extract_region(df):

    return (

        df.Country_Region +

        df.Province_State.fillna("").apply(lambda s: " + " + s if s else s)

    )

    

region_encoder = LabelEncoder().fit(extract_region(train_df))

train_df["region_id"] = region_encoder.transform(extract_region(train_df))

test_df["region_id"] = region_encoder.transform(extract_region(test_df))



region_metadata["region_id"] = region_encoder.transform(extract_region(region_metadata))

region_metadata.drop(["Province_State", "Country_Region"], axis=1, inplace=True)
stats_df = train_df[["Fatalities", "ConfirmedCases", "region_id"]].groupby("region_id").sum()

stats_df["fatalities_to_cases"] = stats_df.Fatalities - stats_df.ConfirmedCases

stats_df.drop(["Fatalities", "ConfirmedCases"], axis=1, inplace=True)

train_df = train_df.merge(stats_df, on="region_id", how="left")

stats_columns = ["fatalities_to_cases"]
def extract_country_from_region(region):

    if "+" in region:

        country, state = region.split(" + ")

    else:

        country = region

        

    return country



region_to_country = dict(

    zip(

        range(len(region_encoder.classes_)),

        map(extract_country_from_region, region_encoder.classes_)

    )

)
train_df = train_df.merge(region_metadata, on="region_id", how="left")
meta_columns = [

    "density",

    "population",

    "area"

]
categorical_columns = [

    "region_id",

    "lockdown_type"

]
def compute_historical_features(df, target_column, past_horizon, create_target=True):

    

    features = dict()

    

    for col in meta_columns + country_meta_columns:

        features[col] = df[col].unique().item()

        

    for col in geo_location_columns:

        features[col] = df[col].unique().item()

        

    for col in stats_columns:

        features[col] = df[col].unique().item()

        

    features["region_id"] = df["region_id"].unique().item()



    last = -1 if create_target else len(df)

    

    target_values = df[target_column].values

        

    for lag in range(past_horizon):

        features["value_{}_{}".format(target_column, lag)] = target_values[last-lag-1]

        

    features["days_since_lockdown"] = df.days_since_lockdown.max()

    features["lockdown_type"] = df.lockdown_type.values[last-1]

    

    if create_target:

        features[target_column] = df[target_column].values[-1]

        

    return features
def prepare_datasets(

    train_df,

    last_train_date,

    target_column,

    train_start_offset=50,

    step=1,

):



    dates = train_df.Date.unique()

    train_dates = dates[dates <= last_train_date]

    val_dates = dates[dates > last_train_date]



    train_features = []



    train_subdf = train_df[train_df.Date <= last_train_date]

    val_subdf = train_df[train_df.Date > last_train_date]



    for region_id in tqdm(sorted(train_df.region_id.unique()), desc="Making train features"):



        subdf = train_subdf[train_subdf.region_id == region_id]



        for date in train_dates[train_start_offset::step]:

            features = compute_historical_features(

                subdf[subdf.Date <= date],

                target_column=target_column,

                past_horizon=train_start_offset,

                create_target=True

            )

            train_features.append(features)

            

    return pd.DataFrame(train_features), train_subdf, val_subdf
def autoregressive_predict(model, dates, df, target_column, past_horizon):

    

    n_regions = len(region_encoder.classes_)

    

    original_size = len(df)

    future_df = pd.DataFrame({

        "Date": np.repeat(dates, repeats=n_regions),

        "region_id": np.tile(np.arange(n_regions), len(dates))

    })

    future_df["Country_Region"] = future_df.region_id.map(region_to_country)

    

    combined_df = pd.concat([

        df.drop(meta_columns + country_meta_columns + geo_location_columns + stats_columns, axis=1),

        future_df

    ], sort=True)

    combined_df = combined_df.merge(region_metadata, on="region_id", how="left")

    combined_df = combined_df.merge(country_data, on=["Country_Region"], how="left")

    combined_df = combined_df.merge(stats_df, on="region_id", how="left")



    for date in tqdm(dates, "Prediction"):

        date_features = []

        for region_id in range(len(region_encoder.classes_)):

            cond = (combined_df.Date < date) & (combined_df.region_id == region_id)

            features = compute_historical_features(

                combined_df[cond],

                target_column=target_column,

                past_horizon=past_horizon,

                create_target=False

            )

            date_features.append(features)

            

        date_predictions = model.predict(pd.DataFrame(date_features))

        combined_df.loc[combined_df.Date == date, target_column] = date_predictions

        

    return combined_df[["Date", "region_id", target_column]].loc[original_size:].reset_index(drop=True)
def target_metric(actual, predicted):

    return np.sqrt(((actual - predicted) ** 2).mean())
def train_booster(

    train_features_df,

    target_column,

    categorical_columns,

    epochs=20,

    train_start_offset=50,

    val_df=None

):



    train_dataset = lgb.Dataset(

        train_features_df.drop([target_column], axis=1),

        train_features_df[target_column],

        free_raw_data=False,

        categorical_feature=categorical_columns

    )



    params = {

        'boosting_type': 'gbdt',

        'objective': 'rmse',

        'num_leaves': 31,

        'max_depth': 7,

        'learning_rate': 0.01,

        'feature_fraction': 0.5,

        'num_threads': os.cpu_count()

    }



    rounds_per_epoch = 300



    model = None



    for epoch in range(epochs):



        model = lgb.train(

            params,

            train_dataset,

            num_boost_round=rounds_per_epoch,

            valid_sets=[train_dataset],

            valid_names=["train"],

            verbose_eval=rounds_per_epoch // 5,

            init_model=model

        )



        if val_df is not None:

            

            val_dates = val_df.Date.unique()

            

            val_predictions = autoregressive_predict(

                model,

                dates=val_dates,

                df=train_subdf,

                past_horizon=train_start_offset,

                target_column=target_column

            )



            error = target_metric(

                val_subdf.sort_values(by=["Date", "region_id"])[target_column].values,

                val_predictions.sort_values(by=["Date", "region_id"])[target_column].values

            )



            print("Epoch", epoch, "error:", error)

            print()

        

    return model
train_start_offset = 70

epochs = 10

test_dates = test_df.Date[test_df.Date > train_df.Date.max()].unique()

merge_columns = ["Date", "region_id"]



for target_column in ["ConfirmedCases", "Fatalities"]:

    

    print()

    print("Working on column", target_column)

    print()

    

    train_features_df, train_subdf, val_subdf = prepare_datasets(

        train_df,

        # no validation <._.>

        last_train_date=train_df.Date.max(),

        target_column=target_column,

        train_start_offset=train_start_offset

    )



    model = train_booster(

        train_features_df,

        target_column=target_column,

        categorical_columns=categorical_columns,

        epochs=epochs,

        train_start_offset=train_start_offset,

        val_df=None

    )



    test_predictions = autoregressive_predict(

        model,

        dates=test_dates,

        df=train_subdf,

        past_horizon=train_start_offset,

        target_column=target_column

    )



    predicted_test_df = test_df.merge(

        pd.concat([

            train_df[merge_columns + [target_column]],

            test_predictions[merge_columns + [target_column]]

        ], sort=True),

        on=merge_columns,

        how="left"

    )



    submission_df = (

        submission_df

        .drop([target_column], axis=1)

        .merge(

            predicted_test_df[[target_column, "ForecastId"]],

            on=["ForecastId"],

            how="left"

        )

    )

    

    submission_df[target_column] = np.expm1(submission_df[target_column])
submission_df.to_csv("submission.csv", index=False)