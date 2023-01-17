from pandas import read_csv

data = read_csv('../input/glass.csv')

target = "Type"

X = data.drop([target],1)

y = data[target]
data[target] = data[target].apply(str)
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
from sklearn.model_selection import cross_val_score



def get_results(model, X, y):



    import warnings

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        compute = cross_val_score(model, X, y)

        mean = compute.mean()

        std = compute.std()

        return mean, std



from time import time

from pandas import DataFrame

from IPython.display import display



def display_classifier_results(models,X,y):



    output = {}



    for m in models:

        try:

            

            model_name = type(m).__name__

            

            start = time()

            scores = get_results(m,X,y)

            finish = time() - start

            

            time_finished = "%d minutes%2d seconds" % (int(finish / 60), finish % 60) 

            

            row = {"Average Score" : scores[0].round(2), "Standard Deviation" : scores[1].round(2), "Processing Time": time_finished}

            

            output[model_name] = row

            

        except:

            pass



    display(DataFrame(data=output, index=["Average Score", "Standard Deviation", "Processing Time"]).T.sort_values("Average Score", ascending=False))
display_classifier_results(models,X,y)
from pandas import DataFrame



def compare_samples_results(samples, model, target):

    

    table = {}

    

    for key in samples.keys():

        

        X = samples[key].drop(target,1)

        y = samples[key][target]

        

        start = time()

        

        mean, std = get_results(model,X,y)

        

        finish = time() - start

        processing_time = "%d minutes%2d seconds" % (int(finish / 60), finish % 60) 

        

        table[key] = {

            "Mean Accuracy Score" : mean.round(3),

            "Standard Deviation" : std.round(3),

            "Processing Time" : processing_time,

        }

        

    return (DataFrame(table).T)[["Mean Accuracy Score","Standard Deviation","Processing Time"]].sort_values(["Mean Accuracy Score"],ascending=False)
sample_template = {"Control":data}

samples = sample_template.copy()
a = ['RI', 'Na', 'Al', 'Si', 'K', 'Ca']

samples["A"] = data[a+[target]]
compare_samples_results(samples,GradientBoostingClassifier(),target)
samples = sample_template.copy()
samples["A"] = data[a+[target]]
dough = samples["A"].copy()

dough["RI / Na"] = dough["RI"] / dough["Na"]

samples["B"] = dough
dough = samples["A"].copy()

dough["RI * Si"] = dough["RI"] * dough["Si"]

samples["C"] = dough
dough = samples["A"].copy()

dough["Na / RI"] = dough["Na"] / dough["RI"]

samples["D"] = dough
dough = samples["A"].copy()

dough["RI / Na"] = dough["RI"] / dough["Na"]

dough["RI * Si"] = dough["RI"] * dough["Si"]

dough["Na / RI"] = dough["Na"] / dough["RI"]

samples["E"] = dough
dough = samples["A"].copy()

dough["RI / Na"] = dough["RI"] / dough["Na"]

dough["RI * Si"] = dough["RI"] * dough["Si"]

samples["F"] = dough
compare_samples_results(samples,GradientBoostingClassifier(),target)
from pandas import DataFrame, Series

from IPython.display import display

from sklearn.model_selection import StratifiedKFold



def get_cross_validation_mean_score(full_data,category,model,folds):

    

    # === KFold Object === #

    

    splitter = StratifiedKFold(n_splits=folds)

    

    # === Keep Model Template === #

    

    model_copy = model

    

    # === Split full data as feature and label data === #

    

    feature_data = full_data.drop(category,1)

    label_data = full_data[category]

    

    # === Set Up List for Scores === #

    

    scores = []

    

    # === For Every Split, Add Accuracy Score by Label Dictionary to Scores List === #

    

    for train_indices, test_indices in splitter.split(feature_data, label_data):

        

        # === Test Data. Actual Label for Index === #

        

        actuals = full_data.iloc[test_indices][category]

        

        # === Reset to Unfitted Model === #

        

        model = model_copy

        

        # === Prepare Input Data for Fitting === #

        

        feature_data = full_data.iloc[train_indices].drop(category,1)

        label_data = full_data.iloc[train_indices][category]

        

        # === Fit the Data === #

        

        model.fit(feature_data,label_data)

        

        # === Obtain predictions from fitted model === #

        

        predictions = model.predict(full_data.iloc[test_indices].drop(category,1))

        

        # === Get accuracy score by label dictionary, then add to scores list === #

        

        scores += [get_score(actuals,predictions)]

        

    # === Return a mean score by label dictionary === #

        

    mean_score = DataFrame(scores).mean().round(3).to_dict()

        

    return mean_score



def get_score(actuals, predictions):

    

    # === Prepare dictionary for accuracy score for each unique label === #

    

    score_dictionary = {}

    

    # === Set count to 0 for all labels === #

    

    for value in actuals.unique():

        score_dictionary[value] = 0

        

    # === Get total counts of each label in actual series === #

    

    actuals_counts = actuals.value_counts()

    

    # === Convert actuals series into list === #

    

    actuals = actuals.tolist()

    

    # === For every matched item by index in actuals and predictions list, add +1 to their counts === #

    

    for i in range(0,len(actuals)):

        

        if actuals[i] == predictions[i]:

            

            value = actuals[i]

            

            score_dictionary[value] += 1

            

    # === Divide label counts correctly guessed by total actual counts in actuals === #

            

    for key in score_dictionary.keys():

        score_dictionary[key] /= actuals_counts[key]

        

    # === Mean Accuracy === #

        

    score_dictionary["Mean Accuracy"] = Series(score_dictionary).mean()

        

    # === Return a score dictionary for this instance of classification predictions === #

                

    return score_dictionary
def compare_samples_results(samples, model, target):

    

    table = {}

    

    for key in samples.keys():

        

        start = time()

       

        row = get_cross_validation_mean_score(samples[key],target,model,3)

        

        finish = time() - start

        

        processing_time = "%d minutes%2d seconds" % (int(finish / 60), finish % 60) 

        

        row["Processing Time"] = processing_time

        

        table[key] = row

        

    return DataFrame(table).T
samples = sample_template.copy()

samples["Control"].columns
a = ['RI', 'Na', 'Al', 'Si', 'K', 'Ca']

samples["A"] = data[a+[target]]

samples["A"].columns
dough = samples["A"].copy()

dough["RI * Si"] = dough["RI"] * dough["Si"]

samples["C"] = dough

samples["C"].columns
compare_samples_results(samples,GradientBoostingClassifier(),target)