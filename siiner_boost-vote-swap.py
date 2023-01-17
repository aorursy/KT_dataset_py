import sys

import os

import pandas as pd

import numpy as np



# prettier print function

from pprint import pprint



# Three popular gradient boosting libraries

import xgboost

import lightgbm

import catboost



# scikit-learn: Machine Learning in Python

from sklearn.ensemble import VotingRegressor

from sklearn.preprocessing import (

    StandardScaler,

    MinMaxScaler,

    OneHotEncoder,

    FunctionTransformer,

)

from sklearn.compose import make_column_transformer

from sklearn.pipeline import make_pipeline

from sklearn.metrics import make_scorer, mean_squared_error

from sklearn.model_selection import train_test_split, cross_val_score, KFold



# dimension reduction methods (tested but not very useful)

from sklearn.decomposition import PCA, FactorAnalysis, FastICA, NMF





# Swapping drugs A and B doubles observations in the same feature space

def swap_drug_name(col_name):

    """ Swap column names between drug A and B. """

    if col_name.startswith("Drug A"):

        return col_name.replace("Drug A", "Drug B")

    elif col_name.startswith("Drug B"):

        return col_name.replace("Drug B", "Drug A")

    else:

        return col_name





def feature_interaction(data, drug_a_cols):

    """

        Create interaction features between drug A and B

        - Absolute differences

        - Products

        - Sum

        - Minimum

        - Maximum

    """

    drug_b_cols = [swap_drug_name(i) for i in drug_a_cols]



    diff = np.abs(data[drug_a_cols].values - data[drug_b_cols].values)

    prod = data[drug_a_cols].values * data[drug_b_cols].values

    sumv = data[drug_a_cols].values + data[drug_b_cols].values

    minv = (

        np.c_[data[drug_a_cols].values.flatten(), data[drug_b_cols].values.flatten()]

        .min(axis=1)

        .reshape(data[drug_a_cols].shape)

    )

    maxv = (

        np.c_[data[drug_a_cols].values.flatten(), data[drug_b_cols].values.flatten()]

        .max(axis=1)

        .reshape(data[drug_a_cols].shape)

    )



    interact_colnames = [

        "{}_{}".format(i, j.replace("Drug A:", ""))

        for i in ("diff", "prod", "sum", "min", "max")

        for j in drug_a_cols

    ]

    interact_features = pd.DataFrame(

        np.c_[diff, prod, sumv, minv, maxv], columns=interact_colnames, index=data.index

    )



    return interact_features





def load_data(input_dir, add_feature=False):

    """ Loading data from the input folder """



    train = pd.read_csv(os.path.join(input_dir, 'train.csv'), index_col=None)

    test = pd.read_csv(os.path.join(input_dir, 'test.csv'), index_col=None)

    train = train.set_index("Id")

    test = test.set_index("Id")



    if add_feature:

        drug_a_cols = [c for c in train.columns if c.startswith("Drug A:")]



        interact_train = feature_interaction(train, drug_a_cols)

        interact_test = feature_interaction(test, drug_a_cols)



        train = pd.concat(

            [train, interact_train], ignore_index=False, axis=1, sort=False

        )



        test = pd.concat([test, interact_test], ignore_index=False, axis=1, sort=False)



    return train, test





def build_model(

    categorical, discrete, continuous, xgb_params=None, lgb_params=None, cat_params=None

):

    """ data transformation pipeline and XGBoost regressor """



    # Scale data: [0, 1] for categorical variables, N(1,1) for quantitative variables

    col_transform = make_column_transformer(

        (

            OneHotEncoder(handle_unknown="ignore"),

            categorical[:1],

        ),  # encoding cell lines

        (

            OneHotEncoder(handle_unknown="ignore"),

            categorical[1:],

        ),  # encoding drug pairs

        (MinMaxScaler(), discrete),  # range [0, 1] for discrete variables

        (StandardScaler(), continuous),  # standardise continuous variables

    )



    regressors = []

    if xgb_params is not None:

        # create a XGBoost regressor

        xgb_reg = xgboost.XGBRegressor(**xgb_params)  # passing keyword parameters by **

        regressors.append(xgb_reg)



    if lgb_params is not None:

        # create a LightGBM regressor

        lgb_reg = lightgbm.LGBMRegressor(**lgb_params)

        regressors.append(lgb_reg)



    if cat_params is not None:

        # create a CatBoost regressor

        cat_reg = catboost.CatBoostRegressor(**cat_params)

        regressors.append(cat_reg)



    if len(regressors) == 1:

        regressor = regressors[0]

    elif len(regressors) > 1:

        regressor = VotingRegressor(

            [(str(i + 1), r) for i, r in enumerate(regressors)], n_jobs=1

        )

    else:

        raise (IOError, "Number of regressors should not be zero!")



    ## make a data processing pipeline

    model = make_pipeline(col_transform, regressor)

    return model





def make_submission(predictions, submission_filename="submission.csv"):

    """

        Create a valid submission from test data 

        Save the file as submission_filename.

    """



    submission_df = pd.DataFrame()

    submission_df["Id"] = [

        "TST{:05d}".format(i) for i in range(1, len(predictions) + 1, 1)

    ]

    submission_df["Loewe Synergy"] = predictions

    submission_df.to_csv(submission_filename, index=False)





def train_validate_model(

    train, xgb_params=None, lgb_params=None, cat_params=None, k=5, swap_ab="no"

):

    """ 

        - Load data

        - Fit into pipeline

        - Validate performance

        - Return the regressor

    """



    if k == 1:

        # validate only once

        train, val = train_test_split(train, test_size=0.1, random_state=2000)



    if swap_ab.lower() == "yes":

        # swapping properties of drug A and B, which doubles the training data

        train_swap_ab = train.copy()

        train_swap_ab.columns = [swap_drug_name(i) for i in train.columns]

        train = train.append(train_swap_ab, ignore_index=True, sort=False)

        del train_swap_ab



    # basic statistics about the features

    description = train.describe()



    # drop columns that have no variance

    discard_cols = description.columns[description.loc["std"] == 0]

    train.drop(discard_cols, axis=1, inplace=True)

    description.drop(discard_cols, axis=1, inplace=True)



    # special columns

    cell_drug_cols = ["Cell Line", "Drug A", "Drug B"]

    target_col = "Loewe Synergy"



    # Consider variables with less than 10 unique values as discrete,

    discrete_filter = train[description.columns].round(2).nunique() < 10

    discrete = description.columns[discrete_filter].tolist()



    # Otherwise, continuous features (but drop the target column)

    continuous = description.columns[~discrete_filter].tolist()

    continuous.remove(target_col)



    model = build_model(

        cell_drug_cols, discrete, continuous, xgb_params, lgb_params, cat_params

    )



    X = train.fillna(0)

    y = train[target_col]



    if k > 1:

        # K-fold cross validation

        np.random.seed(2000)

        kfold = KFold(n_splits=k)

        x_ix = X.index.values

        np.random.shuffle(x_ix)



        scores = []

        for train_ix, val_ix in kfold.split(x_ix):

            train_ix = x_ix[train_ix]

            val_ix = x_ix[val_ix]

            model.fit(X.loc[train_ix], y.loc[train_ix])



            mse = mean_squared_error(y.loc[val_ix].values, model.predict(X.loc[val_ix]))

            scores.append(mse)

            print("MSE: {:.3f}".format(mse), file=sys.stderr)



        print("K-fold (K={}) ave.: {}".format(len(scores), np.mean(scores)))

        print("K-fold (K={}) s.d.: {}".format(len(scores), np.std(scores)))



    elif k == 1:

        # validate only once by the splited validation set

        X_train = train.fillna(0)

        y_train = train[target_col]



        model.fit(X_train, y_train)

        mse = mean_squared_error(val[target_col].values, model.predict(val))



        print("MSE: {:.3f}".format(mse))



        # predict twice with swapped drug properties

        val_swap = val.copy()

        val_swap.columns = [swap_drug_name(i) for i in val.columns]

        mse = mean_squared_error(

            val[target_col].values, (model.predict(val) + model.predict(val_swap)) / 2.0

        )



        print("MSE (swapped average): {:.3f}".format(mse))



    return model





if __name__ == "__main__":



    input_dir = '../input/data/'

    train, test = load_data(input_dir, add_feature=False)

    # However, adding interaction features did not show improvement.



    # set XGBoost parameters

    xgb_params = dict(

        n_jobs=8,  # number of CPU

        max_depth=5,  # max depth of each tree

        objective="reg:squarederror",

        booster="gbtree",

        learning_rate=0.1,

        n_estimators=3000,

        random_state=99,

        verbosity=1,

    )



    # set LightGBM parameters

    lgb_params = dict(

        boosting_type="gbdt",

        max_depth=6,

        num_leaves=40,

        learning_rate=0.1,

        n_estimators=5000,

        random_state=99,

        n_jobs=8,

    )



    # set CatBoost parameters

    cat_params = dict(

        task_type="CPU",  # GPU version can be a little faster

        iterations=3000,

        depth=6,

        learning_rate=0.1,

        eval_metric="RMSE",

        leaf_estimation_iterations=10,

        random_state=99,

        thread_count=8,

    )



    # We can disable one or two regressors by setting parameter(s) to None

    xgb_params = None

#     lgb_params = None

    cat_params = None



    pprint(

        dict(

            zip(

                ["XGBoost", "LightGBM", "CatBoost"],

                [xgb_params, lgb_params, cat_params],

            )

        )

    )



    # We create three overfitting regressors and vote by taking average predicts.

    # Using these parameters requires large (~80G) RAM and multiple (32) CPUs.

    

    model = train_validate_model(

        train=train,

        xgb_params=xgb_params,

        lgb_params=lgb_params,

        cat_params=cat_params,

        swap_ab="yes",

        k=1,

    )



    test_swap = test.copy()

    test_swap.columns = [swap_drug_name(i) for i in test.columns]

    final_prediction = (model.predict(test) + model.predict(test_swap)) / 2

    

    make_submission(final_prediction, submission_filename="Boost_Voting_Swap.txt")


