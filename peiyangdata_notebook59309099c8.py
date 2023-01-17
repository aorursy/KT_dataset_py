# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



X_train = pd.read_json("../input/train.json")

X_test = pd.read_json("../input/test.json")


X_train.head()
X_train.shape
X_test.head()
X_test.shape
sample=pd.read_csv("../input/sample_submission.csv")
sample.head()
print(check_output(["ls", "../input/images_sample/"]).decode("utf8"))
import os

import subprocess as sub

from os import listdir

from os.path import isfile, join

onlyfiles = [f for f in listdir('../input/images_sample/6811957/') if isfile(join('../input/images_sample/6811957/', f))]

print (onlyfiles)
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline

img=[]

for i in range (0,5):

    img.append(mpimg.imread('../input/images_sample/6811957/'+onlyfiles[i]))

    plt.imshow(img[i])

    fig = plt.figure()

    a=fig.add_subplot()

    




X_train.dropna(subset = ['interest_level'])

X_train.shape

print (X_train['interest_level'])


grouped = X_train.groupby(['interest_level'])

print (grouped.size())

#probability

print (grouped.size()/len(X_train))
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss



X_train["num_photos"] = X_train["photos"].apply(len)

X_train["num_features"] = X_train["features"].apply(len)

X_train["num_description_words"] = X_train["description"].apply(lambda x: len(x.split(" ")))

X_train["created"] = pd.to_datetime(X_train["created"])

X_train["created_year"] = X_train["created"].dt.year

X_train["created_month"] = X_train["created"].dt.month

X_train["created_day"] = X_train["created"].dt.day

num_feats = ["bathrooms", "bedrooms", "latitude", "longitude", "price",

             "num_photos", "num_features", "num_description_words",

             "created_year", "created_month", "created_day"]

X = X_train[num_feats]

y = X_train["interest_level"]



X_train2, X_val, y_train2, y_val = train_test_split(X, y, test_size=0.3)


X_test["num_photos"] = X_test["photos"].apply(len)

X_test["num_features"] = X_test["features"].apply(len)

X_test["num_description_words"] = X_test["description"].apply(lambda x: len(x.split(" ")))

X_test["created"] = pd.to_datetime(X_test["created"])

X_test["created_year"] = X_test["created"].dt.year

X_test["created_month"] = X_test["created"].dt.month

X_test["created_day"] = X_test["created"].dt.day

X_test2 = X_test[num_feats]
# Train uncalibrated random forest classifier on whole train and validation

# data and evaluate on test data

rfmodel = RandomForestClassifier(n_estimators=300)

rfmodel.fit(X_train2, y_train2)
X_train2.head()
y_val_pred = rfmodel.predict_proba(X_val)

log_loss(y_val, y_val_pred)
#This time use all the train datasets to train model

rfmodel2 = RandomForestClassifier(n_estimators=300)

rfmodel2.fit(X, y)
y_test_pred = rfmodel2.predict_proba(X_test2)
y_test_pred
X.head()
y.head()
X.shape
y.shape
X_test.shape
X_test2.shape
y1 = [0]*len(y)

y2 = [0]*len(y)



type(y2)
type(y)
y_list = y.tolist()

type(y_list)

y_list
for i in range(0, len(y), 1):

    if(y_list[i]=="low"):

        print("low")

        y1[i] = 0

        y2[i] = 0

    if(y_list[i]=="medium"):

        y1[i] = 0

        y2[i] = 1

    if(y_list[i]=="high"):

        print("high")

        y1[i] = 1

        y2[i] = 0

print("Binary coding done.")
y_n = [0]*len(y)

for i in range(0, len(y), 1):

    if(y_list[i]=="low"):

        y_n[i]=1

    if(y_list[i]=="medium"):

        y_n[i]=2

    if(y_list[i]=="high"):

        y_n[i]=3

print("3-level coding done.")



    
len(y_n)
y_n
y1
y2
y1 == y2
y1_a = np.ravel(y1)

y1_a
y2_a = np.ravel(y2)

y2_a
y_n_a = np.array(y_n)
import numpy as np

import pandas as pd

import statsmodels.api as sm

import matplotlib.pyplot as plt

from patsy import dmatrices

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split

from sklearn import metrics

from sklearn.cross_validation import cross_val_score



#This function takes X_train, y_train to train a 3-classes logistic model

#(so y_train must be a categorical response and take 3 values);

#Then the logistic model is applied to X_test and gives the predicted probability for each row in X_test;

def logistic_regression_3classes(X_train, y_train, X_test):

    model_logistic = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=400, multi_class='multinomial', n_jobs=1,

          penalty='l2', random_state=None, solver='newton-cg', tol=0.0001,

          verbose=0, warm_start=False)

    model_logistic = model_logistic.fit(X_train, y_train)

    prob_test = model_logistic.predict_proba(X_test)

    return prob_test


model_logistic = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=400, multi_class='multinomial', n_jobs=1,

          penalty='l2', random_state=None, solver='newton-cg', tol=0.0001,

          verbose=0, warm_start=False)

X.shape
y_n_a.shape
model_logistic = model_logistic.fit(X, y_n_a)
prob_test = model_logistic.predict_proba(X_test2)
prob_test
lst_id = X_test['listing_id']
type(prob_test)
type(lst_id)
p_d_t = np.vstack((lst_id, prob_test[:,2], prob_test[:,1], prob_test[:,0])).T
type(p_d_t)
p_d_t
p_d_t = pd.DataFrame(p_d_t)
p_d_t[0]=p_d_t[0].astype(int)
p_d_t
p_d_t.to_csv('./result0420a.csv', index=False)