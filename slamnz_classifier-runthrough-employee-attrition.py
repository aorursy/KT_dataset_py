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

data = read_csv("../input/MFG10YearTerminationData.csv")
dtype = get_feature_lists_by_dtype(data)



remove = ["EmployeeID","STATUS_YEAR","store_name", "termreason_desc", "termtype_desc"]

counts = [feature for feature in dtype["int64"] if feature not in remove]

to_be_transformed = ["recorddate_key", "birthdate_key", "orighiredate_key", "terminationdate_key"]

categories = [feature for feature in dtype["object"] if feature not in to_be_transformed or feature not in remove]

categories += ["store_name"]
target= ["STATUS"]
data.shape
num = 1000

x1 = data[data[target[0]] == "ACTIVE"].sample(n=num)

x2 =data[data[target[0]] == "TERMINATED"].sample(n=num)

from pandas import concat

data = concat([x1,x2],0)
y = data[target[0]]



from pandas import get_dummies, concat

one_hot_encoded = get_dummies(data[categories].drop(target,1))



X = concat([data[counts],one_hot_encoded], 1)

#X = data[counts]
def get_results(model, X, y):



    import warnings

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        from sklearn.model_selection import cross_val_score

        compute = cross_val_score(model, X, y, cv=10)

        mean = compute.mean()

        std = compute.std()

        return mean, std



def display_classifier_results(X,y):



    models = []



    from xgboost import XGBClassifier

    models += [XGBClassifier()]

    

    from sklearn.neighbors import KNeighborsClassifier

    models += [KNeighborsClassifier()]



    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

    models += [GaussianNB(), MultinomialNB(), BernoulliNB()]



    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier#, VotingClassifier

    models += [RandomForestClassifier(), AdaBoostClassifier(), GradientBoostingClassifier(), ExtraTreesClassifier()]



    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

    models += [LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis()]



    from sklearn.svm import SVC, LinearSVC

    models += [SVC(),LinearSVC()]



    from sklearn.linear_model import SGDClassifier

    models += [SGDClassifier()]



    from sklearn.neighbors.nearest_centroid import NearestCentroid

    models += [NearestCentroid()]



    output = {}



    for m in models:

        try:

            model_name = type(m).__name__

            from time import time

            start = time()

            scores = get_results(m,X,y)

            finish = time() - start

            time_finished = "%d minutes %2d seconds" % (int(finish / 60), finish % 60) 

            row = {"Mean Accuracy" : scores[0], "(+/-)" : scores[1], "Processing Time": time_finished}

            output[model_name] = row

        except:

            pass



    from pandas import DataFrame

    from IPython.display import display



    result = DataFrame(data=output).T

    result = result[["Mean Accuracy", "(+/-)", "Processing Time"]]

    display(result.sort_values("Mean Accuracy", ascending=False))
# === Return Results === #



display_classifier_results(X,y)