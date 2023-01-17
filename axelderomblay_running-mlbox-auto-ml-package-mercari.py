import pandas as pd

pd.read_csv("../input/train.tsv", sep='\t').to_hdf("train.h5", "train")

pd.read_csv("../input/test.tsv", sep='\t').to_hdf("test.h5", "test")
from mlbox.preprocessing import *

from mlbox.optimisation import * 

from mlbox.prediction import *
paths = ["train.h5", "test.h5"] 

target_name = "price"
rd = Reader()

df = rd.train_test_split(paths, target_name)
dft = Drift_thresholder()

df = dft.fit_transform(df)
from sklearn.metrics import mean_squared_error

rmse = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

df["target"] = df["target"].apply(np.log1p)



opt = Optimiser(scoring=rmse)
opt.evaluate(None, df)
space = {

    

        "ne__numerical_strategy" : {"search":"choice", "space":[0]},  

    

        "ce__strategy" : {"search":"choice", "space":["entity_embedding"]},

    

        'est__strategy':{"search":"choice", "space":["LightGBM"]},

        'est__max_depth':{"search":"choice", "space":[10, 11, 12]},

        'est__learning_rate':{"search":"uniform", "space":[0.06, 0.08]},

        'est__subsample':{"search":"uniform", "space":[0.85, 0.92]},

        'est__colsample_bytree':{"search":"uniform", "space":[0.85, 0.93]}

    

        }



params = opt.optimise(space, df, max_evals = 1)  # you can set max_evals > 1 ... 
prd = Predictor()

prd.fit_predict(params, df)
submit = pd.read_csv("../input/sample_submission.csv")

pred = pd.read_csv("save/"+target_name+"_predictions.csv")



submit[target_name] = pred[target_name+"_predicted"].apply(lambda x: np.exp(x)-1).values

submit.to_csv("mlbox.csv", index=False)