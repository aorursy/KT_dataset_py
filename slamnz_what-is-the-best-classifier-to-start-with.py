from pandas import read_csv

data = read_csv("../input/xAPI-Edu-Data.csv")
target = "Class"

features = data.drop(target,1).columns
features_by_dtype = {}



for f in features:

    dtype = str(data[f].dtype)

    if dtype not in features_by_dtype.keys():

        features_by_dtype[dtype] = [f]

    else:

        features_by_dtype[dtype] += [f]
keys = iter(features_by_dtype.keys())

k = next(keys)

l = features_by_dtype[k]

categorical_features = l

k = next(keys)

l = features_by_dtype[k]

numerical_features = l
categorical_features, numerical_features

features, target

pass
data[numerical_features].head()
data[categorical_features].head()
from pandas import get_dummies,concat

onehot_encoded_categorical_data = get_dummies(data[categorical_features])



X = onehot_encoded_categorical_data

y = data[target]
def get_results(model, X, y):



    import warnings

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        from sklearn.model_selection import cross_val_score

        compute = cross_val_score(model, X, y)

        mean = compute.mean()

        std = compute.std()

        return mean, std



def display_classifier_results(X,y):



    models = []



    from sklearn.neighbors import KNeighborsClassifier

    models = [KNeighborsClassifier()]

    

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



    from sklearn.neural_network import MLPClassifier

    models += [MLPClassifier(hidden_layer_sizes=(len(X.columns), 2))]

    

    from xgboost import XGBClassifier

    models += [XGBClassifier()]



    output = {}



    for m in models:

        try:

            model_name = type(m).__name__

            scores = get_results(m,X,y)

            row = {"Average Score" : scores[0], "Standard Deviation" : scores[1]}

            output[model_name] = row

        except:

            pass



    from pandas import DataFrame

    from IPython.display import display



    display(DataFrame(data=output).T.round(2).sort_values("Average Score", ascending=False))



display_classifier_results(X,y)