# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pickle

enron_dict = pickle.load(open("/kaggle/input/files-unix/final_project_dataset_unix.pkl", 'rb'))
import matplotlib.pyplot as plt



def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):

    """ convert dictionary to numpy array of features

        remove_NaN = True will convert "NaN" string to 0.0

        remove_all_zeroes = True will omit any data points for which

            all the features you seek are 0.0

        remove_any_zeroes = True will omit any data points for which

            any of the features you seek are 0.0

        sort_keys = True sorts keys by alphabetical order. Setting the value as

            a string opens the corresponding pickle file with a preset key

            order (this is used for Python 3 compatibility, and sort_keys

            should be left as False for the course mini-projects).

        NOTE: first feature is assumed to be 'poi' and is not checked for

            removal for zero or missing values.

    """





    return_list = []



    # Key order - first branch is for Python 3 compatibility on mini-projects,

    # second branch is for compatibility on final project.

    if isinstance(sort_keys, str):

        import pickle

        keys = pickle.load(open(sort_keys, "rb"))

    elif sort_keys:

        keys = sorted(dictionary.keys())

    else:

        keys = list(dictionary.keys())



    for key in keys:

        tmp_list = []

        for feature in features:

            try:

                dictionary[key][feature]

            except KeyError:

                print("error: key ", feature, " not present")

                return

            value = dictionary[key][feature]

            if value=="NaN" and remove_NaN:

                value = 0

            tmp_list.append( float(value) )



        # Logic for deciding whether or not to add the data point.

        append = True

        # exclude 'poi' class as criteria.

        if features[0] == 'poi':

            test_list = tmp_list[1:]

        else:

            test_list = tmp_list

        ### if all features are zero and you want to remove

        ### data points that are all zero, do that here

        if remove_all_zeroes:

            append = False

            for item in test_list:

                if item != 0 and item != "NaN":

                    append = True

                    break

        ### if any features for a given data point are zero

        ### and you want to remove data points with any zeroes,

        ### handle that here

        if remove_any_zeroes:

            if 0 in test_list or "NaN" in test_list:

                append = False

        ### Append the data point if flagged for addition.

        if append:

            return_list.append( np.array(tmp_list) )



    return np.array(return_list)





def targetFeatureSplit( data ):

    """ 

        given a numpy array like the one returned from

        featureFormat, separate out the first feature

        and put it into its own list (this should be the 

        quantity you want to predict)



        return targets and features as separate lists



        (sklearn can generally handle both lists and numpy arrays as 

        input formats when training/predicting)

    """



    target = []

    features = []

    for item in data:

        target.append( item[0] )

        features.append( item[1:] )



    return target, features

print("There are", len(enron_dict.keys()), "members in the Enron Dictionary.")
print("The names are:\n", list(enron_dict.keys()))
enron_dict.pop('TOTAL', 0)
lst = []

for key in enron_dict:

    if enron_dict[key]['salary'] == 'NaN':

        lst.append(key)



for key in lst:

    del(enron_dict[key])
features = ["salary", "bonus"]

data = featureFormat(enron_dict, features)



for point in data:

    plt.scatter(point[0], point[1])



plt.xlabel("salary")

plt.ylabel("bonus")

plt.show()
print("There are", len(enron_dict.keys()), "members in the Enron Dictionary with salaries not NaN.")
POI = [k for k in enron_dict if enron_dict[k]["poi"] == 1]

print(len(POI),  "POI in the dict.")
print(POI)
def dict_to_list(key,normalizer):

    new_list = []



    for i in enron_dict:

        if enron_dict[i][key] == "NaN" or enron_dict[i][normalizer] == "NaN":

            new_list.append(0.)

        elif enron_dict[i][key] >= 0:

            new_list.append(float(enron_dict[i][key]) / float(enron_dict[i][normalizer]))

    return new_list
fraction_from_poi_email = dict_to_list("from_poi_to_this_person", "to_messages")

fraction_to_poi_email = dict_to_list("from_this_person_to_poi", "from_messages")



count = 0

for i in enron_dict:

    enron_dict[i]["fraction_from_poi_email"] = fraction_from_poi_email[count]

    enron_dict[i]["fraction_to_poi_email"] = fraction_to_poi_email[count]

    count += 1



    

features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email"] 

my_dataset = enron_dict
data = featureFormat(my_dataset, features_list)

for point in data:

    from_poi = point[1]

    to_poi = point[2]

    plt.scatter( from_poi, to_poi )

    if point[0] == 1:

        plt.scatter(from_poi, to_poi, color="r", marker="*")

plt.xlabel("fraction of emails this person gets from poi")

plt.show()
from sklearn import preprocessing

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score
features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email", "shared_receipt_with_poi"]



data = featureFormat(my_dataset, features_list)

labels, features = targetFeatureSplit(data)

from sklearn import model_selection

from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)



lst = []
RF_Classifier = RandomForestClassifier(max_depth=15, criterion = 'entropy')

model2 = AdaBoostClassifier(n_estimators=120,base_estimator= RF_Classifier)

model2.fit(features_train, labels_train)

lst.append(accuracy_score(model2.predict(features_test), labels_test))
model3 = LogisticRegression(random_state=0)

model3.fit(features_train, labels_train)

lst.append(accuracy_score(model3.predict(features_test), labels_test))
model4 = make_pipeline(StandardScaler(), SVC(gamma='auto'))

model4.fit(features_train, labels_train)

lst.append(accuracy_score(labels_test, model4.predict(features_test)))
model5 = GaussianNB()

model5.fit(features_train, labels_train)

pred = model5.predict(features_test)

accuracy = accuracy_score(pred, labels_test)

lst.append(accuracy)
print("Max accuracy:", max(lst))