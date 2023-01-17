import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import sys
import pickle
from sklearn import preprocessing
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
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
import sys
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pickle
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
data = pickle.load(open("final_dataset.pkl", "rb"))
data.pop('TOTAL')
def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    return_list = []
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
fig,a=plt.subplots(3,2,squeeze=False,figsize=(15,10))
list1=featureFormat(data,["poi","salary","deferral_payments"])
for y in range(len(list1)):
    if list1[y][0]==True:
        a[0][0].scatter(list1[y][1],list1[y][2],color='r')
    else:
        a[0][0].scatter(list1[y][1],list1[y][2],color='g')
list2=featureFormat(data,["poi","total_payments","loan_advances"])
for z in range(len(list2)):
    if list2[z][0]==True:
        a[0][1].scatter(list2[z][1],list2[z][2],color='r')
    else:
        a[0][1].scatter(list2[z][1],list2[z][2],color='g')
list3=featureFormat(data,["poi","total_stock_value","exercised_stock_options"])
for z in range(len(list3)):
    if list3[z][0]==True:
        a[1][0].scatter(list3[z][1],list3[z][2],color='r')
    else:
        a[1][0].scatter(list3[z][1],list3[z][2],color='g')
list4=featureFormat(data,["poi","salary","loan_advances","deferral_payments"])
for z in range(len(list4)):
    if list4[z][0]==True:
        a[1][1].scatter(list4[z][1],list4[z][2],color='r')
    else:
        a[1][1].scatter(list4[z][1],list4[z][2],color='g')
list5=featureFormat(data,["poi","total_stock_value","restricted_stock"])
for z in range(len(list5)):
  if list5[z][0]==True:
        a[2][0].scatter(list5[z][1],list5[z][2],color='r')
  else:
        a[2][0].scatter(list5[z][1],list5[z][2],color='g')     
        
list6=featureFormat(data,["poi","exercised_stock_options","restricted_stock"])
for z in range(len(list6)):
    if list6[z][0]==True:
        a[2][1].scatter(list6[z][1],list6[z][2],color='r')
    else:
        a[2][1].scatter(list6[z][1],list6[z][2],color='g') 

plt.show()
list2=featureFormat(data,["poi","total_payments","loan_advances"])
for z in range(len(list2)):
    if list2[z][0]==True:
        a[0][1].scatter(list2[z][1],list2[z][2],color='r')
    else:
        a[0][1].scatter(list2[z][1],list2[z][2],color='g')
list3=featureFormat(data,["poi","total_stock_value","exercised_stock_options"])
for z in range(len(list3)):
    if list3[z][0]==True:
        a[1][0].scatter(list3[z][1],list3[z][2],color='r')
    else:
        a[1][0].scatter(list3[z][1],list3[z][2],color='g')
list4=featureFormat(data,["poi","salary","loan_advances","deferral_payments"])
for z in range(len(list4)):
    if list4[z][0]==True:
        a[1][1].scatter(list4[z][1],list4[z][2],color='r')
    else:
        a[1][1].scatter(list4[z][1],list4[z][2],color='g')
list5=featureFormat(data,["poi","total_stock_value","restricted_stock"])
for z in range(len(list5)):
  if list5[z][0]==True:
        a[2][0].scatter(list5[z][1],list5[z][2],color='r')
  else:
        a[2][0].scatter(list5[z][1],list5[z][2],color='g')     
        
list6=featureFormat(data,["poi","exercised_stock_options","restricted_stock"])
for z in range(len(list6)):
    if list6[z][0]==True:
        a[2][1].scatter(list6[z][1],list6[z][2],color='r')
    else:
        a[2][1].scatter(list6[z][1],list6[z][2],color='g') 

plt.show()
def dict_to_list(key,normalizer):
    feature_list=[]

    for i in data:
        if data[i][key]=="NaN" or data[i][normalizer]=="NaN":
            feature_list.append(0.)
        elif data[i][key]>=0:
            feature_list.append(float(data[i][key])/float(data[i][normalizer]))
    return feature_list

fraction_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")
fraction_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")
num = 0
for i in data:
    data[i]["fraction_from_poi_email"]=fraction_from_poi_email[num]
    data[i]["fraction_to_poi_email"]=fraction_to_poi_email[num]
    num += 1
feature_list = ['poi','shared_receipt_with_poi','fraction_from_poi_email','fraction_to_poi_email',"deferral_payments"]
feat = featureFormat(data, feature_list)
labels, features = targetFeatureSplit(feat)
X_train, X_test, Y_train, Y_test = train_test_split(features, labels,test_size=0.2,random_state=42)

#random forest
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(max_depth=6, criterion = 'entropy')
rf.fit(X_train,Y_train)
print("accuracy:",accuracy_score(Y_test, rf.predict(X_test)))