# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import sys

import pickle

from sklearn import preprocessing

from time import time

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

# from sklearn.grid_search import GridSearchCV

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

import xgboost

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



original = "../input/enron-dataset/final_project_dataset.pkl"

destination = "final_dataset.pkl"



content = ''

outsize = 0

with open(original, 'rb') as infile:

    content = infile.read()

with open(destination, 'wb') as output:

    for line in content.splitlines():

        outsize += len(line) + 1

        output.write(line + str.encode('\n'))

data = pickle.load(open("final_dataset.pkl", "rb"))

data.pop('TOTAL') #dictionary
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

    target = []

    features = []

    for item in data:

        target.append( item[0] )

        features.append( item[1:] )



    return target, features



# plt.figure(figsize=(20,6))

f1= featureFormat(data,['poi','salary','bonus'])

f2 = featureFormat(data,['poi','salary','total_payments','deferral_payments'])

f3 = featureFormat(data,['poi','salary','total_payments','deferred_income'])

f4 =  featureFormat(data,['poi','from_this_person_to_poi','from_poi_to_this_person'])

fig,a =  plt.subplots(2,2,squeeze=False,figsize=(17,10))

a[0][1].set(ylim=(0, 20000000))

a[1][0].set(ylim=(0, 20000000))

a[1][1].set(xlim=(0,210), ylim=(0,320))

for key in range(len(f1)):

    if f1[key][0] == True:

         a[0][0].scatter(f1[key][1],f1[key][2],color = 'g')

    else:

        a[0][0].scatter(f1[key][1],f1[key][2],color = 'b')

x = np.arange(0,1000000,0.1)  

a[0][0].plot(x,x**1.07,'g')





n = 0

m = 0

for key in range(len(f2)):

    if f2[key][0] == True:

        m+=1

        if f2[key][3]==0:

             a[0][1].plot(f2[key][1],f2[key][2],'g^')

             n+=1

        else:

            a[0][1].plot(f2[key][1],f2[key][2],'go')

    else:

        if f2[key][3]==0:

            a[0][1].plot(f2[key][1],f2[key][2],'b^')

        else:

            a[0][1].plot(f2[key][1],f2[key][2],'bo')



mm  = 0

nn = 0

for key in range(len(f3)):

    if f3[key][0] == True:

        mm+=1

        if f3[key][3]==0:

             a[1][0].plot(f3[key][1],f3[key][2],'g^')

             nn+=1

        else:

            a[1][0].plot(f3[key][1],f3[key][2],'go')

    else:

        if f3[key][3]==0:

            a[1][0].plot(f3[key][1],f3[key][2],'b^')

        else:

            a[1][0].plot(f3[key][1],f3[key][2],'bo')



for key in range(len(f4)):

    if f4[key][0] == True:

         a[1][1].scatter(f4[key][1],f4[key][2],color = 'g')

    else:

        a[1][1].scatter(f4[key][1],f4[key][2],color = 'b')



plt.show()
def dict_to_list(key,normalizer):

    new_list=[]



    for i in data:

        if data[i][key]=="NaN" or data[i][normalizer]=="NaN":

            new_list.append(0.)

        elif data[i][key]>=0:

            new_list.append(float(data[i][key])/float(data[i][normalizer]))

    return new_list



fraction_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")

fraction_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")

j = 0

for i in data:

    data[i]["fraction_from_poi_email"]=fraction_from_poi_email[j]

    data[i]["fraction_to_poi_email"]=fraction_to_poi_email[j]

    j+=1


col = ['poi','shared_receipt_with_poi','fraction_from_poi_email','fraction_to_poi_email',"deferral_payments","bonus"]

data_array = featureFormat(data, col)

x, y = targetFeatureSplit(data_array)

X_train, X_test, Y_train, Y_test = train_test_split(y, x,test_size=0.3 , random_state = 42)



rf=RandomForestClassifier(max_depth=10, criterion = 'entropy')

rf.fit(X_train,Y_train)

print("Accuracy using RandomForestClassifier:",accuracy_score(Y_test, rf.predict(X_test)))



#USING KNN CLASSIFIER

# for i in range(2,17):

#     neigh = KNeighborsClassifier(n_neighbors=i)

#     neigh.fit(X_train,Y_train)

#     print(accuracy_score(Y_test, neigh.predict(X_test)))

#best accuracy for n=6 neighbours

neigh = KNeighborsClassifier(n_neighbors=6)

neigh.fit(X_train,Y_train)

print("Accuracy using KNN classifier: ",accuracy_score(Y_test, neigh.predict(X_test)))



#USING GAUSSIAN NAIVE BAYES CLASSIFIER

gnb=GaussianNB()

gnb.fit(X_train,Y_train)

print("Accuracy using GaussianNB classifier: ",accuracy_score(Y_test, gnb.predict(X_test)))