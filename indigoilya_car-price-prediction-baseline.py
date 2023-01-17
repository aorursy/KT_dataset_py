import warnings

warnings.filterwarnings("ignore")



import numpy as np

import pandas as pd



from pathlib import Path

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GroupKFold

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import KBinsDiscretizer

from sklearn.impute import SimpleImputer 

from sklearn.impute import MissingIndicator

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor
data_folder = Path("../input/TTiDS20/")

submissions_folder = Path.cwd()



train_df = pd.read_csv(data_folder / "train.csv", index_col=0)

test_df = pd.read_csv(data_folder / "test_no_target.csv", index_col=0)

zipcodes_df = pd.read_csv(data_folder / "zipcodes.csv", index_col=0)



train_df = pd.merge(train_df.reset_index(), zipcodes_df.drop_duplicates("zipcode"), on="zipcode", how="left")

test_df = pd.merge(test_df.reset_index(), zipcodes_df.drop_duplicates("zipcode"), on="zipcode", how="left")
cat_features = ["type", "gearbox", "model", "fuel", "brand", "city"]

cont_missing_features = ["engine_capacity", "damage", "insurance_price", "latitude", "longitude"]

cat_missing_features = ["type", "gearbox", "model", "fuel", "city"]
def mape(y_true, y_pred):

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



def zip_dataframes(*dataframes):

    for idx, dataframe in enumerate(dataframes):

        dataframe["df_order"] = idx

    return pd.concat(dataframes)



def unzip_dataframes(dataframe):

    dataframes = []

    for n in dataframe["df_order"].unique().tolist():

        dataframes.append(dataframe[dataframe["df_order"] == n].drop(columns="df_order"))

    return dataframes

    



def create_submit_df(test_df, preds):

    submit_df = pd.DataFrame({

        "Id": test_df["index"],

        "Predicted": preds,

    })

    return submit_df
def preprocessing(train_df, test_df, funcs):

    train_df = train_df.copy()

    test_df = test_df.copy()

    for func in funcs:

        train_df, test_df = func(train_df, test_df)

    return train_df, test_df
def impute_nan_with_zero(train_df, test_df):

    for cat_feature in cat_features:

        train_df[cat_feature] = train_df[cat_feature].fillna("nan")

        test_df[cat_feature] = test_df[cat_feature].fillna("nan")

    train_df = train_df.fillna(0)

    test_df = test_df.fillna(0)

    return train_df, test_df



def impute_nan(train_df, test_df):

    for cont_missing_feature in cont_missing_features:

        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

        imp.fit(pd.concat([train_df, test_df])[[cont_missing_feature]])

        train_df[cont_missing_feature] = imp.transform(train_df[[cont_missing_feature]])

        test_df[cont_missing_feature] = imp.transform(test_df[[cont_missing_feature]])



    for cat_missing_feature in cat_missing_features:

        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value="nan")



        imp.fit(pd.concat([train_df, test_df])[[cat_missing_feature]])

        train_df[cat_missing_feature] = imp.transform(train_df[[cat_missing_feature]])

        test_df[cat_missing_feature] = imp.transform(test_df[[cat_missing_feature]])

    return train_df, test_df



def drop_columns(train_df, test_df):

    drop_columns = ["index"]

    train_df = train_df.drop(columns=drop_columns)

    test_df = test_df.drop(columns=drop_columns)

    return train_df, test_df



def drop_price_outliers(train_df, test_df):

    upper_bound = np.quantile(train_df.price, 0.95)

    train_df = train_df[train_df.price <= upper_bound]

    return train_df, test_df





def drop_insurance_price_outliers(train_df, test_df):

    upper_bound = np.quantile(train_df.insurance_price, 0.99)

    train_df = train_df[train_df.insurance_price <= upper_bound]

    return train_df, test_df



def fill_insurance_price(train_df, test_df):

    train_df.loc[train_df.insurance_price.isna(), "insurance_price"] = train_df.insurance_price.mean()

    return train_df, test_df

    

def fix_registration_year(train_df, test_df):

    train_df.loc[train_df.registration_year < 100, "is_fixed_reg_year"] = 1.0

    train_df.registration_year = train_df.registration_year.apply(lambda y : 2000 + y if y < 21 else y)

    train_df.registration_year = train_df.registration_year.apply(lambda y : 1900 + y if y < 100 else y)

    

    test_df.loc[test_df.registration_year < 100, "is_fixed_reg_year"] = 1.0

    test_df.registration_year = test_df.registration_year.apply(lambda y : 2000 + y if y < 21 else y)

    test_df.registration_year = test_df.registration_year.apply(lambda y : 1900 + y if y < 100 else y)

    return train_df, test_df



def cat_encode(train_df, test_df):

    for cat_feature in cat_features:

        le = LabelEncoder()

        le.fit(pd.concat([train_df, test_df])[cat_feature])

        train_df[cat_feature] = le.transform(train_df[cat_feature])

        test_df[cat_feature] = le.transform(test_df[cat_feature])

        

    return train_df, test_df



def indicate_missing(train_df, test_df):

    for missing_feature in cont_missing_features+cat_missing_features:

        imp = MissingIndicator(missing_values=np.nan)

        imp.fit(pd.concat([train_df, test_df])[[missing_feature]])

        train_df["is_missing_" + missing_feature] = imp.transform(train_df[[missing_feature]])

        test_df["is_missing_" + missing_feature] = imp.transform(test_df[[missing_feature]])

    return train_df, test_df
def cross_validate(

    model,

    train_df,

    kfold,

    metric,

    preproc_funcs,

    target="price",

    test_df=None,

    log_target=False,

    *args,

    **kwargs

):

    val_scores = []

    test_preds = []

    

    if isinstance(kfold, GroupKFold):

        splits = kfold.split(train_df, groups=kwargs["groups"])

    elif isinstance(kfold, StratifiedKFold):

        target_values = train_df[[target]]

        est = KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='quantile')

        stratify_on = est.fit_transform(target_values).T[0]

        splits = kfold.split(train_df, stratify_on)

    else:

        splits = kfold.split(train_df)



    for idx, (tr_idx, val_idx) in enumerate(splits):

        tr_df = train_df.iloc[tr_idx]

        val_df = train_df.iloc[val_idx]

        

        if test_df is not None:

            tr_df, zip_df = preprocessing(tr_df, zip_dataframes(val_df, test_df), preproc_funcs)

            val_df, ts_df = unzip_dataframes(zip_df)

        else:

            tr_df, val_df = preprocessing(tr_df, val_df, preproc_funcs)

        

        x_tr = tr_df.drop(columns=target).values

        y_tr = tr_df[target].values

        x_val = val_df.drop(columns=target).values

        y_val = val_df[target].values

        

        if log_target:

            y_tr = np.log(y_tr)

            y_val = np.log(y_val)

        

        model.fit(x_tr, y_tr)

        preds = model.predict(x_val)

        

        preds = np.exp(preds) if log_target else preds

        y_val = np.exp(y_val) if log_target else y_val

        

        fold_score = metric(y_val, preds)

        val_scores.append(fold_score)

        

        print(f"fold {idx+1} score: {fold_score}")



        if test_df is not None:

            x_ts = ts_df.drop(columns=target).values

            test_fold_preds = model.predict(x_ts)

            test_fold_preds = np.exp(test_fold_preds) if log_target else test_fold_preds

            test_preds.append(test_fold_preds)

            

    print(f"mean score: {np.mean(val_scores)}")

    print(f"score variance: {np.var(val_scores)}")



    if test_df is not None:

        return val_scores, test_preds

    

    return val_scores
# model = XGBRegressor(

#     random_state=42,

#     n_estimators=500,

#     max_depth=5,

#     objective="reg:gamma"

# )



# model = CatBoostRegressor(

#     random_state=42,

#     depth=10,

#     loss_function="MAE",

#     cat_features=[1, 3, 5, 7, 8, 12],

#     verbose=False,

# )
model = LGBMRegressor(

    random_state=42,

    objective='mape',

    num_leaves=100,

    feature_fraction=0.9,

    max_depth=-1,

    learning_rate=0.03,

    num_iterations=1300,

    subsample=0.5,

)



kfold = StratifiedKFold(n_splits=4, random_state=42, shuffle=True)

preproc_funcs = [

    indicate_missing,

    impute_nan_with_zero,

    drop_columns,

    cat_encode,

]



val_scores, test_preds = cross_validate(

    model, 

    train_df,

    kfold,

    mape,

    preproc_funcs,

    test_df=test_df,

    log_target=True,

)
submit_df = create_submit_df(test_df, np.mean(test_preds, axis=0))

submit_df.to_csv(submissions_folder / "lgbm-logtarget-stratkfold.csv", index=False)

submit_df