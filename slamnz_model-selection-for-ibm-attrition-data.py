from pandas import read_csv

data = read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")
target = "Attrition"

feature_by_dtype = {}

for c in data.columns:

    

    if c == target: continue

    

    data_type = str(data[c].dtype)

    

    if data_type not in feature_by_dtype.keys():

         feature_by_dtype[data_type] = [c]

    else:

        feature_by_dtype[data_type].append(c)



objects = feature_by_dtype["object"]

remove = ["Over18"]

categorical_features = [f for f in objects if f not in remove]

int64s = feature_by_dtype["int64"]

count_features = []

for i in [i for i in int64s if len(data[i].unique()) < 20 and i not in remove]:

    count_features.append(i)

remove.append("StandardHours")

remove.append("EmployeeCount")

count_features += ["TotalWorkingYears", "YearsAtCompany", "HourlyRate"]

remove.append("EmployeeNumber")

numerical_features = [i for i in int64s if i not in remove]

features = categorical_features + numerical_features



for c in categorical_features:

    data[c] = data[c].apply(str)



# Global variables

features, target, categorical_features, numerical_features, count_features

pass
from pandas import get_dummies,concat

onehot_encoded_categorical_data = get_dummies(data[categorical_features])



X = concat([data[numerical_features], onehot_encoded_categorical_data], axis=1)

y = data[target]
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

    models += [MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30, 2), random_state=1)]



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