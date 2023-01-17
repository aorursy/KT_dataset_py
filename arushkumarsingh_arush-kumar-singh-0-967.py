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
email = pd.read_csv('../input/enron-email-dataset/emails.csv')
email
import os
print(os.listdir("../input/dataset"))
for i in os.listdir("../input/dataset"):
    try:
        df = pd.read_pickle('../input/dataset/'+i, compression=None)
        print(i, ", SHAPE:", df.shape, ", SIZE: {:,} bytes".format(sys.getsizeof(df)))
        del df
    except Exception as e:
        print(i, "Error loading file", repr(e))
"""
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py 
"""
original = "/kaggle/input/dataset/final_project_dataset.pkl"
destination = "final_project_dataset_unix.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))
dataset = pickle.load(open("final_project_dataset_unix.pkl", "rb"))
data = np.asanyarray(dataset)
data
"""" A general tool for converting data from the
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
"""""

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
xx1 = featureFormat(dataset,['poi','salary','bonus'])
xx2 = featureFormat(dataset,['poi','salary','total_payments','deferral_payments'])
xx3 = featureFormat(dataset,['poi','salary','bonus','deferral_payments'])
xx4 = featureFormat(dataset,['poi','salary','bonus','deferred_income'])
xx5 = featureFormat(dataset,['poi','salary','total_payments','deferred_income'])
xx6 = featureFormat(dataset,['poi','salary','expenses'])
xx2
a = plt.figure(figsize = (8,8))
for i in range(len(xx1)):
    if xx1[i][0] == 1:
        plt.scatter(xx1[i][1],xx1[i][2],color = 'g')
    else:
        plt.scatter(xx1[i][1],xx1[i][2],color = 'r')
        
            
plt.xlabel('Salary')
plt.ylabel('Bonus')













plt.ylim(-1000,10000000)
plt.xlim(-1000,1000000)







plt.show
a = plt.figure(figsize = (8,8))

for i in range(len(xx2)):
    if xx2[i][0] == 1:
        if xx2[i][3] == 0:
            plt.plot(xx2[i][1],xx2[i][2],'g^')
        else: 
            plt.plot(xx2[i][1],xx2[i][2],'go')
    else:
        if xx2[i][3] == 0:
            plt.plot(xx2[i][1],xx2[i][2],'r^')
        else:
            plt.plot(xx2[i][1],xx2[i][2],'ro')
plt.ylim(-1000,10000000)
plt.xlim(-1000,1000000)           
plt.xlabel('Salary')
plt.ylabel('Total_payments') 
plt.show
a = plt.figure(figsize = (8,8))
for i in range(len(xx3)):
    if xx3[i][0] == 1:
        if xx3[i][3] == 0:
            plt.plot(xx3[i][1],xx3[i][2],'g^')
        else: 
            plt.plot(xx3[i][1],xx3[i][2],'go')
    else:
        if xx3[i][3] == 0:
            plt.plot(xx3[i][1],xx3[i][2],'r^')
        else:
            plt.plot(xx3[i][1],xx3[i][2],'ro')
            
plt.ylim(-1000,10000000)
plt.xlim(-1000,1000000)             
plt.xlabel('Salary')
plt.ylabel('Bonus')
plt.show
a = plt.figure(figsize = (8,8))
for i in range(len(xx4)):
    if xx4[i][0] == 1:
        if xx4[i][3] == 0:
            plt.plot(xx4[i][1],xx4[i][2],'g^')
        else: 
            plt.plot(xx4[i][1],xx4[i][2],'go')
    else:
        if xx4[i][3] == 0:
            plt.plot(xx4[i][1],xx4[i][2],'r^')
        else:
            plt.plot(xx4[i][1],xx4[i][2],'ro')
            
plt.ylim(-1000,10000000)
plt.xlim(-1000,1000000)             
plt.xlabel('Salary')
plt.ylabel('Bonus')
plt.show
a = plt.figure(figsize = (8,8))
for i in range(len(xx5)):
    if xx5[i][0] == 1:
        if xx5[i][3] == 0:
            plt.plot(xx5[i][1],xx5[i][2],'g^')
        else: 
            plt.plot(xx5[i][1],xx5[i][2],'go')
    else:
        if xx5[i][3] == 0:
            plt.plot(xx5[i][1],xx5[i][2],'r^')
        else:
            plt.plot(xx5[i][1],xx5[i][2],'ro')
plt.ylim(-1000,10000000)
plt.xlim(-1000,1000000)             
plt.xlabel('Salary')
plt.ylabel('Total_payments')
plt.show
a = plt.figure(figsize = (8,8))
for i in range(len(xx6)):
    if xx6[i][0] == 1:
        plt.scatter(xx6[i][1],xx6[i][2],color = 'g')
    else:
        plt.scatter(xx6[i][1],xx6[i][2],color = 'r')
plt.xlabel('Salary')
plt.ylabel('Expenses')         
        
plt.ylim(-1000,1000000)
plt.xlim(-1000,1000000)         


plt.show
def dict_to_list(key,normalizer):
    feature_list=[]

    for i in dataset:
        if dataset[i][key]=="NaN" or dataset[i][normalizer]=="NaN":
            feature_list.append(0.)
        elif dataset[i][key]>=0:
            feature_list.append(float(dataset[i][key])/float(dataset[i][normalizer]))
    return feature_list

fraction_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")
fraction_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")
p = 0
for i in dataset:
    dataset[i]["fraction_from_poi_email"]=fraction_from_poi_email[p]
    dataset[i]["fraction_to_poi_email"]=fraction_to_poi_email[p]
    p=p+1
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
feature_list = ['poi','shared_receipt_with_poi','fraction_from_poi_email','fraction_to_poi_email',"deferral_payments"]
data_array = featureFormat(dataset, feature_list)
labels, features = targetFeatureSplit(data_array)
X_train, X_test, Y_train, Y_test = train_test_split(features, labels,test_size=0.3,random_state=42)
clf1 = GaussianNB()
clf1.fit(X_train,Y_train)
pred = clf1.predict(X_test)
acc = metrics.accuracy_score(pred,Y_test)
print("Accuracy by GaussianNB classifier: ",acc)

clf2=RandomForestClassifier()
clf2.fit(X_train,Y_train)
pred2 = clf2.predict(X_test)
acc2 = metrics.accuracy_score(pred2,Y_test)
print("Accuracy by RandomForestClassifier:",acc2)
clf3 = KNeighborsClassifier(n_neighbors=6)
clf3.fit(X_train,Y_train)
pred3 = clf3.predict(X_test)
acc3 = metrics.accuracy_score(pred3, Y_test)
print("Accuracy by KNN classifier: ",acc3)
clf4 = tree.DecisionTreeClassifier()
clf4 = clf4.fit(X_train, Y_train)
pred4 = clf4.predict(X_test)
acc = metrics.accuracy_score(pred4,Y_test)
print("Accuracy by DecesionTree classifier: ",acc3)
pickle.dump(clf2, open("my_classifier.pkl", "wb") )
pickle.dump(dataset, open("my_dataset.pkl", "wb") )
pickle.dump(feature_list, open("my_feature_list.pkl", "wb") )



