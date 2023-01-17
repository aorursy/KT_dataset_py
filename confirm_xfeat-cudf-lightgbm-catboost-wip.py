import sys

!cp ../input/rapids/rapids.0.14.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
!pip install -q https://github.com/pfnet-research/xfeat/archive/master.zip
import pandas as pd

from sklearn.model_selection import KFold

import xfeat

import cudf



import catboost as cat

import lightgbm as lgb





xfeat.utils.cudf_is_available()
PATH_TRAIN_CSV = "../input/bnp-paribas-cardif-claims-management/train.csv.zip"

PATH_TEST_CSV = "../input/bnp-paribas-cardif-claims-management/test.csv.zip"

PATH_SUB_CSV = "../input/bnp-paribas-cardif-claims-management/sample_submission.csv.zip"



USECOLS = [

    "v10", "v12", "v14", "v21", "v22", "v24", "v30", "v31", "v34", "v38",

    "v40", "v47", "v50", "v52", "v56", "v62", "v66", "v72", "v75", "v79",

    "v91", "v112", "v113", "v114", "v129", "target"

]





def preload():

    # Convert dataset into feather format.

    xfeat.utils.compress_df(pd.concat([

        pd.read_csv(PATH_TRAIN_CSV),

        pd.read_csv(PATH_TEST_CSV),

    ], sort=False)).reset_index(drop=True)[USECOLS].to_feather(

        "../working/train_test.ftr")





preload()
pd.read_feather("../working/train_test.ftr").head()
print("(1) Save numerical features")

xfeat.SelectNumerical().fit_transform(

    pd.read_feather("../working/train_test.ftr")

).reset_index(drop=True).to_feather("../working/feature_num_features.ftr")
pd.read_feather("../working/feature_num_features.ftr").head()
print("(2) Categorical encoding using label encoding: 13 features")

xfeat.Pipeline([

    xfeat.SelectCategorical(), xfeat.LabelEncoder(output_suffix="")]

).fit_transform(

    pd.read_feather("../working/train_test.ftr")

).reset_index(drop=True).to_feather("../working/feature_1way_label_encoding.ftr")
pd.read_feather("../working/feature_1way_label_encoding.ftr").head()
def target_encoding(path_source_dataframe, path_output_dataframe):

    df_train_test = pd.read_feather(path_source_dataframe)

    df_train_test.loc[:, "target"] = pd.read_feather("../working/train_test.ftr")["target"]

    df_train = df_train_test.dropna(subset=["target"]).copy()

    df_test = df_train_test.loc[df_train_test["target"].isnull()].copy()



    fold = KFold(n_splits=5, shuffle=True, random_state=111)

    encoder = xfeat.TargetEncoder(fold=fold, target_col="target", output_suffix="")

    df_train = encoder.fit_transform(cudf.from_pandas(df_train))

    df_test = encoder.transform(cudf.from_pandas(df_test))



    pd.concat([df_train.to_pandas(), df_test.to_pandas()], sort=False).drop(

        "target", axis=1).reset_index(drop=True).to_feather(path_output_dataframe)





print("(2') Target encoding of categorical variables")

target_encoding("../working/feature_1way_label_encoding.ftr",

                "../working/feature_1way_label_encoding_with_te.ftr")
pd.read_feather("../working/feature_1way_label_encoding_with_te.ftr").head()
print("(3) 2-order combination of categorical features: 78 features (13 * 12 / 2 = 78)")

xfeat.Pipeline([

    xfeat.SelectCategorical(),

    xfeat.ConcatCombination(drop_origin=True, r=2),

    xfeat.LabelEncoder(output_suffix=""),

]).fit_transform(pd.read_feather("../working/train_test.ftr")).reset_index(

    drop=True

).to_feather(

    "../working/feature_2way_label_encoding.ftr"

)
pd.read_feather("../working/feature_2way_label_encoding.ftr").head()
print("(4) 3-order combination of categorical features")

# Use `include_cols=` kwargs to reduce the total count of combinations.

# 66 features (12 * 11 / 2 = 66)

xfeat.Pipeline([

    xfeat.SelectCategorical(),

    xfeat.ConcatCombination(drop_origin=True, include_cols=["v22"], r=3),

    xfeat.LabelEncoder(output_suffix=""),

]).fit_transform(pd.read_feather("../working/train_test.ftr")).reset_index(

    drop=True

).to_feather(

    "../working/feature_3way_including_v22_label_encoding.ftr"

)





print("(5) Convert numerical to categorical using round: 12 features")

df_rnum = (

    xfeat.Pipeline(

        [

            xfeat.SelectNumerical(),

            xfeat.LambdaEncoder(

                lambda x: str(x)[:-2],

                output_suffix="_rnum",

                exclude_cols=["target"],

            ),

        ]

    )

    .fit_transform(pd.read_feather("../working/train_test.ftr"))

    .reset_index(drop=True)

)

df_rnum.to_feather("../working/feature_round_num.ftr")

rnum_cols = [col for col in df_rnum.columns if col.endswith("_rnum")]

xfeat.Pipeline([xfeat.LabelEncoder(output_suffix="")]).fit_transform(

    pd.read_feather("../working/feature_round_num.ftr")[rnum_cols]

).reset_index(drop=True).to_feather("../working/feature_round_num_label_encoding.ftr")





print("(6) 2-order Arithmetic combinations.")

xfeat.Pipeline(

    [

        xfeat.SelectNumerical(),

        xfeat.ArithmeticCombinations(

            exclude_cols=["target"], drop_origin=True, operator="+", r=2,

        ),

    ]

).fit_transform(pd.read_feather("../working/train_test.ftr")).reset_index(

    drop=True

).to_feather(

    "../working/feature_arithmetic_combi2.ftr"

)





print("(7) Add more combinations: 11-order concat combinations.")

xfeat.Pipeline(

    [

        xfeat.SelectCategorical(),

        xfeat.ConcatCombination(drop_origin=True, include_cols=["v22"], r=11),

        xfeat.LabelEncoder(output_suffix=""),

    ]

).fit_transform(pd.read_feather("../working/train_test.ftr")).reset_index(

    drop=True

).to_feather(

    "../working/feature_11way_including_v22_label_encoding.ftr"

)





print("(3') Target encoding of categorical variables")

target_encoding("../working/feature_2way_label_encoding.ftr",

                "../working/feature_2way_label_encoding_with_te.ftr")



print("(4') Target encoding of categorical variables")

target_encoding("../working/feature_3way_including_v22_label_encoding.ftr",

                "../working/feature_3way_including_v22_label_encoding_with_te.ftr")



print("(5') Target encoding of categorical variables")

target_encoding("../working/feature_round_num_label_encoding.ftr",

                "../working/feature_round_num_label_encoding_with_te.ftr")



print("(7') Target encoding of categorical variables")

target_encoding("../working/feature_11way_including_v22_label_encoding.ftr",

                "../working/feature_11way_including_v22_label_encoding_with_te.ftr")
!ls -lha ../working/
import catboost as cat





def catboost_model():

    print("Load numerical features")

    df_num = pd.concat(

        [

            pd.read_feather("../working/feature_num_features.ftr"),

            pd.read_feather("../working/feature_arithmetic_combi2.ftr"),

        ],

        axis=1,

    )

    y_train = df_num["target"].dropna()

    df_num.drop(["target"], axis=1, inplace=True)



    print("Load categorical features")

    df = pd.concat(

        [

            pd.read_feather("../working/feature_1way_label_encoding.ftr"),

            pd.read_feather("../working/feature_2way_label_encoding.ftr"),

            pd.read_feather("../working/feature_3way_including_v22_label_encoding.ftr"),

            pd.read_feather("../working/feature_round_num_label_encoding.ftr"),

            pd.read_feather("../working/feature_11way_including_v22_label_encoding.ftr"),

        ],

        axis=1,

    )

    cat_cols = df.columns.tolist()

    df = pd.concat([df, df_num], axis=1)



    print("Fit")

    params = {

        "loss_function": "Logloss",

        "eval_metric": "Logloss",

        "learning_rate": 0.03,

        "iterations": 3000,

        "l2_leaf_reg": 3,

        "random_seed": 432013,

        "subsample": 0.66,

        "od_type": "Iter",

        "rsm": 0.2,

        "depth": 6,

        "border_count": 128,

    }

    model = cat.CatBoostClassifier(**params)

    train_data = cat.Pool(

        df.iloc[: y_train.shape[0]], label=y_train, cat_features=cat_cols

    )

    fit_model = model.fit(train_data, verbose=30)



    # Predict

    y_pred = fit_model.predict_proba(df.iloc[y_train.shape[0] :])

    submission = pd.read_csv("../input/bnp-paribas-cardif-claims-management/sample_submission.csv.zip")

    submission.loc[:, "PredictedProb"] = y_pred[:, 1]

    submission.to_csv("../working/solution_cat.csv", index=False)





LIGHTGBM_BASE_PARAMS = {

    "objective": "binary",

    "metric": "binary_logloss",

    "verbosity": 1,

    "boosting_type": "gbdt",

    "num_leaves": 32,

    "feature_fraction": 0.8,

    "bagging_fraction": 1.0,

    "bagging_freq": 0,

    "learning_rate": 0.03,

    "max_bin": 255,

    "seed": 1,

    "min_data_in_leaf": 20,

}

LIGHTGBM_BASE_FIT_PARAMS = {

    "verbose_eval": 30,

    "num_boost_round": 3000,

}





def lightgbm_model():

    print("Load numerical features")

    df_num = pd.concat(

        [

            pd.read_feather("../working/feature_num_features.ftr"),

            pd.read_feather("../working/feature_arithmetic_combi2.ftr"),

        ],

        axis=1,

    )

    y_train = df_num["target"].dropna()

    df_num.drop(["target"], axis=1, inplace=True)



    print("Load categorical features")

    df = pd.concat(

        [

            pd.read_feather("../working/feature_1way_label_encoding_with_te.ftr"),

            pd.read_feather("../working/feature_2way_label_encoding_with_te.ftr"),

            pd.read_feather("../working/feature_3way_including_v22_label_encoding_with_te.ftr"),

            pd.read_feather("../working/feature_round_num_label_encoding_with_te.ftr"),

            pd.read_feather("../working/feature_11way_including_v22_label_encoding_with_te.ftr"),

        ],

        axis=1,

    )

    cat_cols = df.columns.tolist()

    df = pd.concat([df, df_num], axis=1)



    X_train, X_test = df.values[:y_train.shape[0]], df.values[y_train.shape[0]:]

    dtrain = lgb.Dataset(X_train, y_train)

    

    fit_params = LIGHTGBM_BASE_FIT_PARAMS.copy()

    fit_params["valid_sets"] = [dtrain]



    bst = lgb.train(LIGHTGBM_BASE_PARAMS, dtrain, **fit_params)

    y_pred = bst.predict(X_test)



    # Predict

    submission = pd.read_csv("../input/bnp-paribas-cardif-claims-management/sample_submission.csv.zip")

    submission.loc[:, "PredictedProb"] = y_pred

    submission.to_csv("../working/solution_lgb.csv", index=False)





lightgbm_model()   

catboost_model()





# Linear blending

y1 = pd.read_csv("../working/solution_cat.csv")

y2 = pd.read_csv("../working/solution_lgb.csv")



df_sub = pd.read_csv(PATH_SUB_CSV)

df_sub.loc[:, "PredictedProb"] = 0.6 * y1.PredictedProb.values + 0.4 * y2.PredictedProb.values

df_sub[["ID", "PredictedProb"]].to_csv("../working/solution_blend.csv", index=False)