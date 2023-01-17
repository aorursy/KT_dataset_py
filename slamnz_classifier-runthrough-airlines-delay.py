from pandas import read_csv

data = read_csv("../input/DelayedFlights.csv")
data = data.drop("Unnamed: 0",1)
target = ["Cancelled"]

leaky_features = ["Year", "Diverted", "ArrTime", "ActualElapsedTime", "AirTime", "ActualElapsedTime", "AirTime", "ArrDelay", "TaxiIn", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay","LateAircraftDelay", "CancellationCode"]

features = [x for x in data.columns if (x != target[0]) & (x not in leaky_features) & (len(data[x].unique().tolist()) > 1)]
where = data["Month"].isin([10,11,12])

data = data[where]
def get_dtypes(data,features):

    output = {}

    for f in features:

        dtype = str(data[f].dtype)

        if dtype not in output.keys(): output[dtype] = [f]

        else: output[dtype] += [f]

    return output
dtypes = get_dtypes(data,features)
categories = ["Month", "DayOfWeek", "DayofMonth"]

categories += dtypes["object"]

numerics = [i for i in dtypes["int64"] if i not in categories]

numerics += dtypes["float64"]
data[categories].head()
data[numerics].head()
for numeric in numerics: data[numeric] = data[numeric].fillna(0)
categories.remove("TailNum")
cancelled = data[data[target[0]] == 1]

not_cancelled = data[data[target[0]] == 0]
from pandas import concat



data = concat([cancelled, not_cancelled.sample(n=len(cancelled))],0)
from pandas import get_dummies



one_hot_encoded = get_dummies(data[categories].fillna("Unknown"))

X = concat([one_hot_encoded, data[numerics].fillna(0)],1)

y = data[target[0]]
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
from itertools import groupby

from numpy import nan



def split_text(text):

    

    sequence = [''.join(unit) for split_point, unit in groupby(text, str.isalpha)]

    

    if len(sequence) < 3:

        

        if sequence[0].isalpha():

            

            sequence += [str(nan)]

            

        else:

            

            sequence = [str(nan)] + sequence

            

        

    return tuple(sequence)



def split_tailnum(data, series):

    

    TailNum_0 = []

    TailNum_1 = []

    TailNum_2 = []

    

    round_by_first_2 = lambda x: round(int(x),2 - len(x))

    

    for value in series: 

        splits = split_text(value)

        TailNum_0 += [splits[0]]

        TailNum_1 += [round_by_first_2(splits[1])]

        TailNum_2 += [splits[2]]

    

    data["TailNum_0"] = TailNum_0

    data["TailNum_1"] = TailNum_1

    data["TailNum_2"] = TailNum_2

    data["TailNum_2_has_AA"] = data["TailNum_2"].apply(lambda x: 1 if "AA" in x else 0)

    

    return data

    

data = split_tailnum(data, data["TailNum"])



categories += ["TailNum_0","TailNum_1","TailNum_2", "TailNum_2_has_AA"]
cancelled = data[data[target[0]] == 1]

not_cancelled = data[data[target[0]] == 0]

data = concat([cancelled, not_cancelled.sample(n=len(cancelled))],0)

one_hot_encoded = get_dummies(data[categories].fillna("Unknown"))

X = concat([one_hot_encoded, data[numerics].fillna(0)],1)

y = data[target[0]]
display_classifier_results(X,y)