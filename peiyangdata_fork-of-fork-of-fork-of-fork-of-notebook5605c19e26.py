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
y_val_pred = rfmodel.predict_proba(X_val)

log_loss(y_val, y_val_pred)
#This time use all the train datasets to train model

rfmodel2 = RandomForestClassifier(n_estimators=300)

rfmodel2.fit(X, y)
y_test_pred = rfmodel2.predict_proba(X_test2)
y_test_pred
X_train2.head()
y.head()
y.head()





    

    
type(y)
length = len(y)

length
y1 = [0]*length

y2 = [0]*length



type(y2)
len(y2) == length
len(y1) == length
type(y)
y_list = y.tolist()

type(y_list)

y_list
for i in range(0, length, 1):

    if(y_list[i]=="low"):

        print("low")

        y1[i] = 0

        y2[i] = 0

    if(y_list[i]=="medium"):

        print("medium")

        y1[i] = 0

        y2[i] = 1

    if(y_list[i]=="high"):

        print("high")

        y1[i] = 1

        y2[i] = 0

print("Binary coding done.")
y1
y2
y1 == y2
y1_a = np.ravel(y1)

y1_a
y2_a = np.ravel(y2)

y2_a
import numpy as np

import pandas as pd

import statsmodels.api as sm

import matplotlib.pyplot as plt

from patsy import dmatrices

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split

from sklearn import metrics

from sklearn.cross_validation import cross_val_score
model1 = LogisticRegression()

model1 = model1.fit(X, y1_a)



model1.score(X, y1_a)
model2 = LogisticRegression()

model2 = model2.fit(X, y2_a)



model2.score(X, y2_a)
1-y1_a.mean()
1-y2_a.mean()
model1
model2
X_pyang1 = X.loc[range(9,11,1),:]

X_pyang1
probs1 = model1.predict_proba(X_pyang1)

probs1
probs2 = model2.predict_proba(X_pyang1)

probs2
X_pyang2 = X.dropna()

X_pyang2
probs1_2 = model1.predict_proba(X_pyang2)

probs1_2
probs2_2 = model2.predict_proba(X_pyang2)

probs2_2
type(probs1_2)
p1 = np.array(probs1_2)

p1.shape
xx = np.array(X)

xx.shape
xxx = np.array(X_pyang2)

xxx.shape
xx == xxx
prob_high = p1[:,1]

prob_high
p2 = np.array(probs2_2)



prob_medium = p2[:,1]



prob_medium
prob_low = 1 - prob_high - prob_medium

prob_low
#check if prob_low has negative values

N = len(prob_low)



count = 0

for i in range(0, N, 1):

    if(prob_low[i])<0:

        print("alert: negative probability detected!!")

        count += 1
print(count)
# So 41 records will be "both medium and high level interest"

# If this happens, can we just let prob(low)=0?



prob_low_adjusted = prob_low

prob_low_adjusted[prob_low_adjusted<0]=0



for i in range(0, N, 1):

    if(prob_low_adjusted[i])<0:

        print("alert: negative probability detected!!")
# we have got prob_high, prob_medium, prob_low_adjusted

# concatenate them

prob_low.shape

prob_medium.shape

prob_high.shape

distribution_matrix = np.vstack((prob_low_adjusted, prob_medium, prob_high)).T



distribution_matrix
prob_low_adjusted
model1
model2
X_test.head()
prob_high_test = model1.predict_proba(X_test2)

prob_high_test
prob_medium_test = model2.predict_proba(X_test2)

prob_medium_test
type(prob_high_test)
p_high_test = np.array(prob_high_test)

p_medium_test = np.array(prob_medium_test)
len(p_high_test)
len(X_test2)
len(X_pyang2)
len(p_medium_test)
p_medium_test
p_h_test = p_high_test[:,1]

p_m_test = p_medium_test[:,1]
len(p_h_test)
p_h_test.shape
p_m_test.shape
p_l_test = 1 - (p_h_test + p_m_test)
for i in range(0, len(p_l_test), 1):

    if(p_l_test[i]<0): print("negative prob detected!!")
p_l_test_adjusted = p_l_test

p_l_test_adjusted[p_l_test_adjusted<0]=0

for i in range(0, len(p_l_test_adjusted), 1):

    if(p_l_test_adjusted[i]<0): print("negative prob detected!!")
prob_distribution_test = np.vstack((p_l_test_adjusted, p_m_test, p_h_test)).T
prob_distribution_test
prob_distribution_test.shape
prob_distribution_test.tofile('C:\999_peiyang_disk\002_ds_project\001_kaggle\001_rental_listing_inquiry\prediction_test.csv',sep=',',format='%10.5f')
prob_dis_t = np.vstack((p_h_test, p_m_test, p_l_test)).T
prob_dis_t
prob_dis_t.to_csv("result.csv",inde=False)
type(prob_dis_t)
p_dis_t = pd.DataFrame(prob_dis_t)
type(p_dis_t)
p_dis_t.to_csv("result.csv", index=True)