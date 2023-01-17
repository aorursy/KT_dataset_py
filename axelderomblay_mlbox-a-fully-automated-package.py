from mlbox.preprocessing import *

from mlbox.optimisation import *

from mlbox.prediction import *
paths = ["../input/train.json", "../input/test.json"]

target_name = "interest_level"
rd = Reader()

df = rd.train_test_split(paths, target_name)
df["train"].head()
dft = Drift_thresholder()

df = dft.fit_transform(df)
opt = Optimiser()
opt.evaluate(None, df)
space = {

    

        "ne__numerical_strategy" : {"search":"choice", "space":["mean", 0]},  

    

        "ce__strategy" : {"search":"choice", "space":["label_encoding", "random_projection"]},

    

        'est__strategy':{"search":"choice", "space":["LightGBM"]},    

        'est__max_depth':{"search":"choice", "space":[6,7]},

        'est__learning_rate':{"search":"uniform", "space":[0.01, 0.03]},

        'est__subsample':{"search":"uniform", "space":[0.8, 0.9]},

        'est__colsamplebytree':{"search":"uniform", "space":[0.7, 0.8]}

    

        }



params = opt.optimise(space, df, 2)
prd = Predictor()

prd.fit_predict(params, df)
submit = pd.read_csv("save/"+target_name+"_predictions.csv")[["high", "medium", "low"]]

submit["listing_id"] = df["test"].listing_id.astype(int).values

submit.to_csv("mlbox.csv", index=False)
stck = {

    "stck__base_estimators" : [

        Classifier(strategy="XGBoost"),

        Classifier(strategy="RandomForest"),

        Classifier(strategy="LightGBM")

    ],

    "est__strategy" : "RandomForest",

    "est__max_depth" : 3,

    "est__max_features" : 1.

}
# prd.fit_predict(stck, df)
# submit = pd.read_csv("save/"+target_name+"_predictions.csv")[["high", "medium", "low"]]

# submit["listing_id"] = df["test"].listing_id.astype(int).values

# submit.to_csv("mlbox_stck.csv", index=False)