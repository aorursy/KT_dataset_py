from mlbox.preprocessing import *

from mlbox.optimisation import *

from mlbox.prediction import *
paths = ["../input/train.csv", "../input/test.csv"]

target_name = "target"
rd = Reader(sep = ",")

df = rd.train_test_split(paths, target_name)   #reading and preprocessing (dates, ...)
dft = Drift_thresholder()

df = dft.fit_transform(df)   #removing non-stable features (like ID,...)
def gini(actual, pred, cmpcol = 0, sortcol = 1):

    assert( len(actual) == len(pred) )

    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)

    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]

    totalLosses = all[:,0].sum()

    giniSum = all[:,0].cumsum().sum() / totalLosses

 

    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)

 

def gini_normalized(a, p):

    return np.abs(gini(a, p) / gini(a, a))





opt = Optimiser(scoring = make_scorer(gini_normalized, greater_is_better=True, needs_proba=True), n_folds=2)
space = {

    

        'est__strategy':{"search":"choice",

                                  "space":["LightGBM"]},    

        'est__n_estimators':{"search":"choice",

                                  "space":[700]},    

        'est__colsample_bytree':{"search":"uniform",

                                  "space":[0.77,0.82]},

        'est__subsample':{"search":"uniform",

                                  "space":[0.73,0.8]},

        'est__max_depth':{"search":"choice",

                                  "space":[5,6,7]},

        'est__learning_rate':{"search":"uniform",

                                  "space":[0.008, 0.02]} 

    

        }



params = opt.optimise(space, df, 7)
prd = Predictor()

prd.fit_predict(params, df)
submit = pd.read_csv("../input/sample_submission.csv",sep=',')

preds = pd.read_csv("save/"+target_name+"_predictions.csv")



submit[target_name] =  preds["1.0"].values



submit.to_csv("mlbox.csv", index=False)