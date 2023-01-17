from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *
paths = ["../input/nyc-taxi-trip-duration/train.csv", "../input/nyc-taxi-trip-duration/test.csv"]
target_name = "trip_duration"
time.sleep(30)
rd = Reader(sep = ",")
df = rd.train_test_split(paths, target_name)   #reading and preprocessing (dates, ...)
cols = [u'id', u'starting_street', u'end_street', u'total_distance',u'total_travel_time', u'number_of_steps']
extra_train = pd.read_csv("../input/new-york-city-taxi-with-osrm/fastest_route_train.csv", usecols=cols)
extra_test = pd.read_csv("../input/new-york-city-taxi-with-osrm/fastest_route_test.csv", usecols=cols)

df['train'] = pd.merge(df['train'], extra_train, on ='id', how='left')
df['test'] = pd.merge(df['test'], extra_test, on ='id', how='left')
df['train']["N2"] = ((df['train']["dropoff_longitude"]-df['train']["pickup_longitude"]).apply(lambda x: x**2) + (df['train']["dropoff_latitude"]-df['train']["pickup_latitude"]).apply(lambda x: x**2)).apply(lambda x: np.sqrt(x))
df['test']["N2"] = ((df['test']["dropoff_longitude"]-df['test']["pickup_longitude"]).apply(lambda x: x**2) + (df['test']["dropoff_latitude"]-df['test']["pickup_latitude"]).apply(lambda x: x**2)).apply(lambda x: np.sqrt(x))

df['train']["N1"] = (df['train']["dropoff_longitude"]-df['train']["pickup_longitude"]).apply(lambda x: np.abs(x)) + (df['train']["dropoff_latitude"]-df['train']["pickup_latitude"]).apply(lambda x: np.abs(x))
df['test']["N1"] = (df['test']["dropoff_longitude"]-df['test']["pickup_longitude"]).apply(lambda x: np.abs(x)) + (df['test']["dropoff_latitude"]-df['test']["pickup_latitude"]).apply(lambda x: np.abs(x))

df['train']["pickup_distance_center"] = ((df['train']["pickup_longitude"].mean()-df['train']["pickup_longitude"]).apply(lambda x: x**2) + (df['train']["pickup_latitude"].mean()-df['train']["pickup_latitude"]).apply(lambda x: x**2)).apply(lambda x: np.sqrt(x))
df['test']["pickup_distance_center"] = ((df['test']["pickup_longitude"].mean()-df['test']["pickup_longitude"]).apply(lambda x: x**2) + (df['test']["pickup_latitude"].mean()-df['test']["pickup_latitude"]).apply(lambda x: x**2)).apply(lambda x: np.sqrt(x))
dft = Drift_thresholder()
df = dft.fit_transform(df)   #removing non-stable features (like ID,...)
df['target'] = df['target'].apply(lambda x: np.log1p(x))   #evaluation metric: rmsle

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

opt = Optimiser(scoring = make_scorer(rmse, greater_is_better=False), n_folds=2)
space = {
     
        'est__strategy':{"search":"choice",
                                  "space":["XGBoost"]},    
        'est__n_estimators':{"search":"choice",
                                  "space":[300]},    
        'est__colsample_bytree':{"search":"uniform",
                                  "space":[0.78,0.82]},   
        'est__colsample_bylevel':{"search":"uniform",
                                  "space":[0.78,0.82]},    
        'est__subsample':{"search":"uniform",
                                  "space":[0.82,0.88]},
        'est__max_depth':{"search":"choice",
                                  "space":[10,11]},
        'est__learning_rate':{"search":"choice",
                                  "space":[0.075]} 
    
        }

params = opt.optimise(space, df, 1)  #only 1 iteration because it takes a long time otherwise :) 
prd = Predictor()
prd.fit_predict(params, df)
submit = pd.read_csv("../input/nyc-taxi-trip-duration/sample_submission.csv",sep=',')
preds = pd.read_csv("save/"+target_name+"_predictions.csv")

submit[target_name] =  preds[target_name+"_predicted"].apply(lambda x: np.exp(x)-1).values

submit.to_csv("mlbox.csv", index=False)