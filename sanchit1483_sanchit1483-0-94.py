# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle

import matplotlib.pyplot as plt

import sklearn

import sklearn.metrics as metrics



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

print(os.listdir("/kaggle/input/input-data"))
original = "/kaggle/input/input-data/final_project_dataset.pkl"

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

""" 

    A general tool for converting data from the

    dictionary format to an (n x k) python list that's 

    ready for training an sklearn algorithm



    n--no. of key-value pairs in dictonary

    k--no. of features being extracted



    dictionary keys are names of persons in dataset

    dictionary values are dictionaries, where each

        key-value pair in the dict is the name

        of a feature, and its value for that person



    In addition to converting a dictionary to a numpy 

    array, you may want to separate the labels from the

    features--this is what targetFeatureSplit is for



    so, if you want to have the poi label as the target,

    and the features you want to use are the person's

    salary and bonus, here's what you would do:



    feature_list = ["poi", "salary", "bonus"] 

    data_array = featureFormat( data_dictionary, feature_list )

    label, features = targetFeatureSplit(data_array)



    the line above (targetFeatureSplit) assumes that the

    label is the _first_ item in feature_list--very important

    that poi is listed first!

"""





import numpy as np



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

    new_list=[]



    for i in final_data:

        if final_data[i][key]=="NaN" or final_data[i][normalizer]=="NaN":

            new_list.append(0.)

        elif final_data[i][key]>=0:

            new_list.append(float(final_data[i][key])/float(final_data[i][normalizer]))

    return new_list



fraction_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")

fraction_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")

j = 0

for i in final_data:

    final_data[i]["fraction_from_poi_email"]=fraction_from_poi_email[j]

    final_data[i]["fraction_to_poi_email"]=fraction_to_poi_email[j]

    j+=1
from sklearn import metrics

from sklearn.model_selection import train_test_split



col = ['poi','shared_receipt_with_poi','fraction_from_poi_email','fraction_to_poi_email',"deferral_payments","bonus"]

data_array = featureFormat(final_data, col)

x, y = targetFeatureSplit(data_array)

X_train, X_test, Y_train, Y_test = train_test_split(y, x,test_size=0.3 , random_state = 42)
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split



test1 = GaussianNB()

test1.fit(X_train,Y_train)

pred = test1.predict(X_test)

acc1 = metrics.accuracy_score(pred,Y_test)

print("ACCURACY");

print("GaussianNB classifier: ",acc1)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



test2=RandomForestClassifier()

test2.fit(X_train,Y_train)

pred2 = test2.predict(X_test)

acc2 = metrics.accuracy_score(pred2,Y_test)

print("ACCURACY")

print("RandomForestClassifier:",acc2)
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier



test3 = KNeighborsClassifier(n_neighbors=6)

test3.fit(X_train,Y_train)

pred3 = test3.predict(X_test)

acc3 = metrics.accuracy_score(pred3, Y_test)

print("ACCURACY")

print("KNN classifier: ",acc3)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



test4 = LogisticRegression(C=0.01, solver='liblinear')

test4.fit(X_train,Y_train)

pred4 = test4.predict(X_test)

acc4 = metrics.accuracy_score(pred4, Y_test)

print("ACCURACY")

print("Logistic Regression : ",acc4)