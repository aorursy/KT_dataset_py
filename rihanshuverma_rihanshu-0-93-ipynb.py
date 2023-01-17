# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import sys
import os
from sklearn.metrics import recall_score
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print (os.path.join(dirname, filename))

sys.path.append('/kaggle/input/enrondataset/')
from feature_format import featureFormat as ft
from feature_format import targetFeatureSplit as tfs

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
real = "/kaggle/input/enrondataset/final_project_dataset.pkl"
final = "final_project_dataset_unix.pkl"

c = ''
outsize = 0
with open(real, 'rb') as data:
    c = data.read()
with open(final, 'wb') as output:
    for i in c.splitlines():
        outsize += len(i) + 1
        output.write(i + str.encode('\n'))
final_data = pickle.load(open("/kaggle/input/enrondataset/final_project_dataset_unix.pkl", 'rb') )

final_data.pop('TOTAL')
p1 = ft(final_data, ['poi','salary','total_payments'])

print("The number of people in the dataset:",len(final_data))
print(list(final_data.keys())[0],"\n",final_data[list(final_data.keys())[0]])

for z in range(len(p1)):
        if p1[z][0]==True:
            plt.scatter(p1[z][1],p1[z][2],color = 'b')
        else:
            plt.scatter(p1[z][1],p1[z][2],color = 'g')

plt.ylabel('total_payments')
plt.xlabel('salary')   
plt.show()
plt.figure(figsize=(10,10))
p2=ft(final_data,["poi","total_payments","loan_advances"])
for z in range(len(p2)):
    if p2[z][0]==True:
        plt.scatter(p2[z][1],p2[z][2],color='b')
    else:
        plt.scatter(p2[z][1],p2[z][2],color='g')
plt.ylabel('loan_advances')
plt.xlabel('total_payments')   
plt.show()
plt.figure(figsize=(10,10))        
p3=ft(final_data,["poi","total_stock_value","exercised_stock_options"])
for z in range(len(p3)):
    if p3[z][0]==True:
        plt.scatter(p3[z][1],p3[z][2],color='b')
    else:
        plt.scatter(p3[z][1],p3[z][2],color='g')
plt.ylabel('exercised_stock_options')
plt.xlabel('total_stock_value')   
plt.show()
plt.figure(figsize=(10,10))        
p4=ft(final_data,["poi","salary","loan_advances","deferral_payments"])
for z in range(len(p4)):
    if p4[z][0]==True:
        plt.scatter(p4[z][1],p4[z][2],color='b')
    else:
        plt.scatter(p4[z][1],p4[z][2],color='g')
plt.ylabel('loan_advances')
plt.xlabel('salary')   
plt.show()
plt.figure(figsize=(10,10))        


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
final_feature_list = ['poi','shared_receipt_with_poi','fraction_from_poi_email','fraction_to_poi_email',"deferral_payments"]
k = ft(final_data, final_feature_list)
labels, features = tfs(k)
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size = 0.15)
reg = LogisticRegression()
x1=reg.fit(X_train,Y_train)
print(accuracy_score(Y_test, x1.predict(X_test)))
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
x2=clf.fit(X_train,Y_train)
print(accuracy_score(Y_test, x2.predict(X_test)))
from sklearn import svm
clf2=svm.SVC()
x3=clf2.fit(X_train,Y_train)
print(accuracy_score(Y_test, x3.predict(X_test)))
pickle.dump(final_data, open("my_dataset.pkl", "wb") )
pickle.dump(final_feature_list, open("my_feature_list.pkl", "wb") )
pickle.dump(x1, open("my_classifier.pkl", "wb") )