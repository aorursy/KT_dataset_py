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
import os

import matplotlib.pyplot as plt

import sys

import pickle

from sklearn import preprocessing

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

original = "/kaggle/input/dataset/final_project_dataset.pkl"

destination = "final_dataset.pkl"



content = ''

outsize = 0

with open(original, 'rb') as infile:

    content = infile.read()

with open(destination, 'wb') as output:

    for line in content.splitlines():

        outsize += len(line) + 1

        output.write(line + str.encode('\n'))

        

final_data = pickle.load(open("final_dataset.pkl", "rb") )
#Remove Outlier

final_data.pop('TOTAL')


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

        keys = dictionary.keys()



    for key in keys:

        tmp_list = []

        for feature in features:

            try:

                dictionary[key][feature]

            except KeyError:

                print ("error: key ", feature, " not present")

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
def dict_to_list(key,normalizer):

    feature_list=[]



    for i in final_data:

        if final_data[i][key]=="NaN" or final_data[i][normalizer]=="NaN":

            feature_list.append(0.)

        elif final_data[i][key]>=0:

            feature_list.append(float(final_data[i][key])/float(final_data[i][normalizer]))

    return feature_list



mail_from_poi_fraction=dict_to_list("from_poi_to_this_person","to_messages")

mail_to_poi_fraction=dict_to_list("from_this_person_to_poi","from_messages")

num = 0

for i in final_data:

    final_data[i]["mail_from_poi_fraction"]=mail_from_poi_fraction[num]

    final_data[i]["mail_to_poi_fraction"]=mail_to_poi_fraction[num]

    num += 1
features_list = ['poi','mail_from_poi_fraction','mail_to_poi_fraction','shared_receipt_with_poi','deferral_payments','deferred_income']

data = featureFormat(final_data, features_list, sort_keys = True)

Y,X = targetFeatureSplit(data)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
#GaussianNB

clf1=GaussianNB()

clf1.fit(X_train,Y_train)

pred1=clf1.predict(X_test)

print("Accuracy for GaussianNB: ",accuracy_score(pred1,Y_test))
#SVM

clf2=SVC()

clf2.fit(X_train,Y_train)

pred2=clf2.predict(X_test)

print("Accuracy for SVM: ",accuracy_score(pred2,Y_test))
#AdaBoost

clf3=AdaBoostClassifier()

clf3.fit(X_train,Y_train)

pred3=clf3.predict(X_test)

print("Accuracy for AdaBoost: ",accuracy_score(pred3,Y_test))
#RandomForest

clf4=RandomForestClassifier()

clf4.fit(X_train,Y_train)

pred4=clf4.predict(X_test)

print("Accuracy for RandomForest: ",accuracy_score(pred4,Y_test))
#parameter tuning for random forest

from sklearn.model_selection import GridSearchCV

params={'n_estimators':[100,200],'criterion':('gini','entropy'),'max_features':('auto','sqrt'),'max_depth':[5,10]}

clf5=GridSearchCV(RandomForestClassifier(),params)

clf5.fit(X_train,Y_train)

pred5=clf5.predict(X_test)

print("Accuracy for RandomForest tuned: ",accuracy_score(pred5,Y_test))
