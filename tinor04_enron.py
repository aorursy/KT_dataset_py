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
import sys

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score



import pickle

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score





enron_dict = pickle.load(open("/kaggle/input/file-unix/final_project_dataset_unix.pkl", 'rb') )



import numpy as np



def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):







    return_list = []



    # Key order - first branch is for Python 3 compatibility on mini-projects,

    # second branch is for compatibility on final project.

    if isinstance(sort_keys, str):

        import pickle

        keys = pickle.load(open(sort_keys, "rb"))

    elif sort_keys:

        keys = sorted(dictionary.keys())

    else:

        keys = dictionary.keys()



    for key in keys:

        tmp_list = []

        for feature in features:

            try:

                dictionary[key][feature]

            except KeyError:

                

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





    target = []

    features = []

    for item in data:

        target.append( item[0] )

        features.append( item[1:] )



    return target, features



print(len(enron_dict))
print (enron_dict.keys())
#check for outliers

features = ["salary", "bonus"]



enron_dict.pop('TOTAL', 0)

data = featureFormat(enron_dict, features)



# remove NAN from dataset



outliers = []

for key in enron_dict:

    val = enron_dict[key]['salary']

    if val == 'NaN':

        continue

    outliers.append((key, int(val)))



outliers_final = (sorted(outliers,key=lambda x:x[1],reverse=True)[:4])

# print top 4 salaries

print (outliers_final)
def dict_to_list(key,normalizer):

    new_list=[]



    for i in enron_dict:

        if enron_dict[i][key]=="NaN" or enron_dict[i][normalizer]=="NaN":

            new_list.append(0.)

        elif enron_dict[i][key]>=0:

            new_list.append(float(enron_dict[i][key])/float(enron_dict[i][normalizer]))

    return new_list





fraction_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")

fraction_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")



### insert new features into data_dict

count=0

for i in enron_dict:

    enron_dict[i]["fraction_from_poi_email"]=fraction_from_poi_email[count]

    enron_dict[i]["fraction_to_poi_email"]=fraction_to_poi_email[count]

    count +=1



features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email"]    

    ### store to my_dataset for easy export below

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
features_list = ["poi", "salary", "bonus", "fraction_from_poi_email", "fraction_to_poi_email",

                 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred',

                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',

                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']

data = featureFormat(my_dataset, features_list)





labels, features = targetFeatureSplit(data)



# split the data

from sklearn import model_selection

from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.1, random_state=42)









from sklearn.tree import DecisionTreeClassifier



clf = DecisionTreeClassifier()

clf.fit(features_train,labels_train)

score = clf.score(features_test,labels_test)

pred= clf.predict(features_test)

print ('accuracy', score)

features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email", "shared_receipt_with_poi"]



# try Naive Bayes for prediction





clf = GaussianNB()

clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

accuracy = accuracy_score(pred,labels_test)

print (accuracy)
RF_Classifier = RandomForestClassifier(max_depth=15, criterion = 'entropy')

model = AdaBoostClassifier(n_estimators=120,base_estimator= RF_Classifier)

model.fit(features_train, labels_train)

print(accuracy_score(labels_test, model.predict(features_test)))
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC





clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

clf.fit(features_train, labels_train)

print(accuracy_score(labels_test, clf.predict(features_test)))
model_3 = LogisticRegression(random_state=0)

model_3.fit(features_train, labels_train)

print(accuracy_score(labels_test, model_3.predict(features_test)))