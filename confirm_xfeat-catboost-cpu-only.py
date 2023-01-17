!pip install -q https://github.com/pfnet-research/xfeat/archive/master.zip
import pandas as pd



USECOLS = [

    "v10", "v12", "v14", "v21", "v22", "v24", "v30", "v31",

    "v34", "v38", "v40", "v47", "v50", "v52", "v56", "v62",

    "v66", "v72", "v75", "v79", "v91", "v112", "v113", "v114", "v129",

]





def preload():

    # Download CSV files first.

    # `$ kaggle competitions download -c bnp-paribas-cardif-claims-management`

    pd.concat([

        pd.read_csv("../input/bnp-paribas-cardif-claims-management/train.csv.zip"),

        pd.read_csv("../input/bnp-paribas-cardif-claims-management/test.csv.zip"),

    ], sort=False).reset_index(drop=True)[USECOLS + ["target"]].to_feather("../working/train_test.ftr")





preload()
from xfeat import SelectNumerical





print("(1) Save numerical features")

SelectNumerical().fit_transform(pd.read_feather("../working/train_test.ftr")).reset_index(

    drop=True

).to_feather("../working/feature_num_features.ftr")
from xfeat import SelectCategorical, LabelEncoder, Pipeline





print("(2) Categorical encoding using label encoding: 13 features")

Pipeline([SelectCategorical(), LabelEncoder(output_suffix="")]).fit_transform(

    pd.read_feather("../working/train_test.ftr")

).reset_index(drop=True).to_feather("../working/feature_1way_label_encoding.ftr")
from xfeat import SelectCategorical, ConcatCombination





print("(3) 2-order combination of categorical features: 78 features (13 * 12 / 2 = 78)")

Pipeline(

    [

        SelectCategorical(),

        ConcatCombination(drop_origin=True, r=2),

        LabelEncoder(output_suffix=""),

    ]

).fit_transform(pd.read_feather("../working/train_test.ftr")).reset_index(

    drop=True

).to_feather(

    "../working/feature_2way_label_encoding.ftr"

)
print("(4) 3-order combination of categorical features")

# Use `include_cols=` kwargs to reduce the total count of combinations.

# 66 features (12 * 11 / 2 = 66)

Pipeline(

    [

        SelectCategorical(),

        ConcatCombination(drop_origin=True, include_cols=["v22"], r=3),

        LabelEncoder(output_suffix=""),

    ]

).fit_transform(pd.read_feather("../working/train_test.ftr")).reset_index(

    drop=True

).to_feather(

    "../working/feature_3way_including_v22_label_encoding.ftr"

)
from xfeat import SelectNumerical, LambdaEncoder





print("(5) Convert numerical to categorical using round: 12 features")

df_rnum = (

    Pipeline(

        [

            SelectNumerical(),

            LambdaEncoder(

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

Pipeline([LabelEncoder(output_suffix="")]).fit_transform(

    pd.read_feather("../working/feature_round_num.ftr")[rnum_cols]

).reset_index(drop=True).to_feather("../working/feature_round_num_label_encoding.ftr")
from xfeat import ArithmeticCombinations





print("(6) 2-order Arithmetic combinations.")

Pipeline(

    [

        SelectNumerical(),

        ArithmeticCombinations(

            exclude_cols=["target"], drop_origin=True, operator="+", r=2,

        ),

    ]

).fit_transform(pd.read_feather("../working/train_test.ftr")).reset_index(

    drop=True

).to_feather(

    "../working/feature_arithmetic_combi2.ftr"

)
print("(7) Add more combinations: 11-order concat combinations.")

Pipeline(

    [

        SelectCategorical(),

        ConcatCombination(drop_origin=True, include_cols=["v22"], r=11),

        LabelEncoder(output_suffix=""),

    ]

).fit_transform(pd.read_feather("../working/train_test.ftr")).reset_index(

    drop=True

).to_feather(

    "../working/feature_11way_including_v22_label_encoding.ftr"

)
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

    submission.to_csv("../working/solution_small_1k.csv", index=False)

    

    

catboost_model()