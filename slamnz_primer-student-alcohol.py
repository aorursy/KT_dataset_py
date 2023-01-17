def get_feature_lists_by_dtype(data):

    features = data.columns.tolist()

    output = {}

    for f in features:

        dtype = str(data[f].dtype)

        if dtype not in output.keys(): output[dtype] = [f]

        else: output[dtype] += [f]

    return output



def show_uniques(data,features):

    for f in features:

        if len(data[f].unique()) < 30:

            print("%s: count(%s) %s" % (f,len(data[f].unique()),data[f].unique()))

        else:

            print("%s: count(%s) %s" % (f,len(data[f].unique()),data[f].unique()[0:10]))



def show_all_uniques(data):

    dtypes = get_feature_lists_by_dtype(data)

    for key in dtypes.keys():

        print(key + "\n")

        show_uniques(data,dtypes[key])

        print()
from pandas import read_csv

data = read_csv("../input/student-mat.csv")
data.head()
data.shape
data.columns
show_all_uniques(data)
features_by_dtype = get_feature_lists_by_dtype(data)
categorical_features = features_by_dtype["object"]
count_features = features_by_dtype["int64"]
count_features, categorical_features

pass
target = ["absences"]



y = data[target]



from pandas import get_dummies,concat

onehot_encoded_categorical_data = get_dummies(data[categorical_features])

X = concat([data[count_features], onehot_encoded_categorical_data], axis=1)



X.drop(target,1, inplace=True)
from numpy import log1p

def squared_logarithmic_error(y_true, y_pred):

    return (log1p(y_pred) - log1p(y_true)) ** 2

def mean_squared_logarithmic_error(y_true, y_pred):

    calculation = squared_logarithmic_error(y_true, y_pred)

    return calculation.sum() / len(calculation)

def root_mean_squared_logarithmic_error(y_true, y_pred):

    return mean_squared_logarithmic_error(y_true, y_pred) ** 0.5
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score



def score_row(actuals,predictions):



    parameters = {"y_true" : actuals,

                 "y_pred" : predictions}



    #score_functions = [explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score, root_mean_squared_logarithmic_error]

    score_functions = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_logarithmic_error]

    

    output = {}



    for func in score_functions:

        output[str(func.__name__)] = func(**parameters) 



    return output
from sklearn.model_selection import KFold

from sklearn.ensemble import GradientBoostingRegressor



def cross_val_score(model, data, features, target_feature):



    iterations = []



    splits = 10

    splitter = KFold(n_splits=splits, random_state=0)

    i = iter(range(0,splits))

    score_rows = []



    for train, test in splitter.split(data):



        training_set = data.iloc[train]

        testing_set = data.iloc[test]



        model.fit(training_set[features],training_set[target_feature])

        iterations += [model]



        predictions = model.predict(testing_set[features])

        actuals = testing_set[target_feature]



        # === Score Metrics ===



        score_rows += [score_row(actuals,predictions)]

        

    return score_rows



from IPython.display import display

from pandas import DataFrame



def display_mean_scores(model, data, features, target):

    print(type(model).__name__)

    display(DataFrame(cross_val_score(model,data,features,target)).mean())

    

from pandas import options

def display_cv_scores(model, data, features, target):

    options.display.float_format = '{:,.3f}'.format

    display(DataFrame(cross_val_score(model,data,features,target)).round(2))



from time import time

from pandas import Series

    

def regressor_runthrough(regressors, data, features, target_feature):

    results = {}

    for r in regressors:

        key = type(r).__name__

        try:

            start = time()

            

            unit = DataFrame(cross_val_score(r,data,features,target_feature)).mean()

            

            finished = time() - start

            

            unit = unit.append(Series([finished], index=["Total Processing Time"]))

            

            results[key] = unit

            

        except:

            pass

            #print(key + " failed.")

    return DataFrame(results).T
regressors = []



from sklearn.svm import SVR, LinearSVR, NuSVR

regressor = SVR()

regressors.append(regressor)

regressor = LinearSVR()

regressors.append(regressor)

regressor = NuSVR()

regressors.append(regressor)



from sklearn.linear_model import HuberRegressor, PassiveAggressiveRegressor, RANSACRegressor, SGDRegressor,TheilSenRegressor

regressors += [HuberRegressor(), PassiveAggressiveRegressor()]



from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor

regressor = KNeighborsRegressor()

regressors.append(regressor)

regressor = RadiusNeighborsRegressor()

regressors.append(regressor)



from sklearn.gaussian_process import GaussianProcessRegressor

regressor = GaussianProcessRegressor()

regressors.append(regressor)



from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()

regressors.append(regressor)



from sklearn.tree import ExtraTreeRegressor

regressor = ExtraTreeRegressor()

regressors.append(regressor)



from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor

regressor = AdaBoostRegressor()

regressors.append(regressor)

regressor = BaggingRegressor()

regressors.append(regressor)

regressor = ExtraTreesRegressor()

regressors.append(regressor)

regressor = RandomForestRegressor()

regressors.append(regressor)

regressor = GradientBoostingRegressor()

regressors.append(regressor)



from xgboost import XGBRegressor

regressors += [XGBRegressor()]
full_data = X.copy()

full_data[target] = data[target]

results = regressor_runthrough(regressors, full_data, X.columns.tolist(), target[0])
from pandas import options

options.display.float_format = "{:.2f}".format

results.sort_values("mean_absolute_error", ascending=True)
target = ["G3"]



y = data[target]



from pandas import get_dummies,concat

onehot_encoded_categorical_data = get_dummies(data[categorical_features])

X = concat([data[count_features], onehot_encoded_categorical_data], axis=1)



X.drop(target,1, inplace=True)



full_data = X.copy()

full_data[target] = data[target]

results = regressor_runthrough(regressors, full_data, X.columns.tolist(), target[0])
options.display.float_format = "{:.2f}".format

results.sort_values("mean_absolute_error", ascending=True)