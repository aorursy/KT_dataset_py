from pandas import read_csv

raw_data = read_csv("../input/Dataset_spine.csv")
# === Rename The Columns === #



column_names = ("pelvic_incidence",

"pelvic_tilt",

"lumbar_lordosis_angle",

"sacral_slope",

"pelvic_radius",

"degree_spondylolisthesis",

"pelvic_slope",

"Direct_tilt",

"thoracic_slope",

"cervical_tilt",

"sacrum_angle",

"scoliosis_slope")



# === Rename === #



rename = {}

for i in range(0,12):

    temp = "Col" + str(i+1)

    rename[temp] = column_names[i]



renamed_data = raw_data.rename(columns = rename)
data = renamed_data.drop("Unnamed: 13",1)

target = "Class_att"

features = [feature for feature in data.columns if feature != target]
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

        

    mean_score = DataFrame(scores).mean().to_dict()

        

    return mean_score
def display_classifier_results(full_data,category,models,folds):



    output = {}



    for m in models:

        try:

            model_name = type(m).__name__

            row = get_cross_validation_mean_score(full_data,category,m,folds)

            output[model_name] = row

        except:

            pass



    from pandas import DataFrame

    from IPython.display import display



    display(DataFrame(data=output, index = ["Abnormal","Normal","Mean Accuracy"]).T.round(2).sort_values("Mean Accuracy", ascending=False))
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



from sklearn.neighbors.nearest_centroid import NearestCentroid

models += [NearestCentroid()]



from xgboost import XGBClassifier

models += [XGBClassifier()]
display_classifier_results(data, target, models, 10)