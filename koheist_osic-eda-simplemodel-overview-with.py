import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

import glob

import re

import cv2
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

train_df
import pydicom



def plot_pixel_array(dataset, figsize=(5,5)):

    plt.figure(figsize=figsize)

    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)

    plt.show()



file_path = "../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/1.dcm"

dataset = pydicom.dcmread(file_path)

plot_pixel_array(dataset)
def extract_num(s, p, ret=0):

    search = p.search(s)

    if search:

        return int(search.groups()[0])

    else:

        return ret
filepath = []

ID = "ID00007637202177411956430"



for file in glob.glob("../input/osic-pulmonary-fibrosis-progression/train/"+ ID +"/*.dcm"):

    filepath.append(file)

    

p = re.compile(ID +"/"+"(\d+)")

filepath = sorted(filepath, key=lambda s: extract_num(s, p, float('inf'))) #画像を数字順にsort
fig = plt.figure(figsize=(16,7))



for i in range(18):

    plt.subplot(3, 6, i+1)

    file_path = filepath[i]

    dataset = pydicom.dcmread(file_path)

    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)

    plt.title(file_path[77:])

    plt.tick_params(labelbottom=False,

                    labelleft=False,

                    labelright=False,

                    labeltop=False)
train_df.loc[train_df.Patient == ID]
Patient_list = list(train_df.Patient.unique())

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))



a = 0

b = 0

c = 0



for ID in Patient_list:

    grp = train_df.loc[train_df.Patient == ID]

    grp = grp[["Weeks","FVC", "SmokingStatus"]]



    if grp.iloc[0, 2] == "Currently smokes" and a <= 10:

        ax1.plot(grp.Weeks, grp.FVC, marker="o", color="red")

        ax1.set_title("Currently smokes") 

        a = a + 1

    elif grp.iloc[0, 2] == "Ex-smoker" and b <= 10:

        ax2.plot(grp.Weeks, grp.FVC, marker="x", color="green")

        ax2.set_title("Ex-smoker") 

        b = b + 1

    elif grp.iloc[0, 2] == "Never smoked" and c <= 10:

        ax3.plot(grp.Weeks, grp.FVC, marker="s", color="blue")

        ax3.set_title("Never smoked")

        c = c + 1

    else:

        pass
Week = np.arange(-12, 134)

train_df2 = pd.DataFrame(Week, columns = ["Weeks"])

train_df2.insert(1, 'FVC', np.nan)

train_df2.insert(2, 'Percent', np.nan)

train_df2.insert(3, 'Age', np.nan)

train_df2.insert(4, 'Sex', np.nan)

train_df2.insert(5, 'SmokingStatus', np.nan)



train_id = train_df.loc[train_df.Patient == Patient_list[1]]

train_id = train_id.reset_index()



for i, D in enumerate(train_id.Weeks):

    D = D + 12

    train_df2.at[D, "FVC"] = train_id.FVC[i]

    train_df2.at[D, "Percent"] = train_id.Percent[i]



train_df2.loc[:, "Age"] = train_id.Age[0]

train_df2.loc[:, "Sex"] = train_id.Sex[0]

train_df2.loc[:, "SmokingStatus"] = train_id.SmokingStatus[0]

    

train_df2 = train_df2.interpolate('linear', order=2, limit_direction='both')

train_df2
plt.figure(figsize=(18,6))

grp = train_df2



plt.xlabel("Weeks")

plt.ylabel("FVC")

plt.plot(grp.Weeks, grp.FVC, marker="x")

plt.plot(train_id.Weeks, train_id.FVC, marker="o", markersize=8)

plt.figure(figsize=(18,6))

grp = train_df2



plt.xlabel("Weeks")

plt.ylabel("Percent")

plt.plot(grp.Weeks, grp.Percent, marker="^")

plt.plot(train_id.Weeks, train_id.Percent, marker="o", markersize=8)
Week = np.arange(-12, 134)



def train_layer (ID_N):

    train_df2   = pd.DataFrame(Week, columns = ["Weeks"])

    train_df_Y  = pd.DataFrame(Week, columns = ["Weeks"])

    train_df_Y.insert(1, 'FVC', np.nan)

    train_df2.insert(1, 'Percent', np.nan)

    train_df2.insert(2, 'Age', np.nan)

    train_df2.insert(3, 'Sex_Male', 0)

    train_df2.insert(4, 'Sex_Female', 0)

    train_df2.insert(5, 'Currently smokes', 0)

    train_df2.insert(6, 'Ex-smoker', 0)

    train_df2.insert(7, 'Never smoked', 0)



    train_id = train_df.loc[train_df.Patient == Patient_list[ID_N]]

    train_id = train_id.reset_index()



    for i, D in enumerate(train_id.Weeks):

        D = D + 12

        if D <= 133:

            train_df_Y.at[D, "FVC"] = train_id.FVC[i]

            train_df2.at[D, "Percent"] = train_id.Percent[i]



    train_df2.loc[:, "Age"] = train_id.Age[0]



    if train_id.Sex[0] == "Male":

        train_df2.loc[:, "Sex_Male"] = 1

    else:

        train_df2.loc[:, "Sex_Female"] = 1

    

    if train_id.SmokingStatus[0] == "Currently smokes":

        train_df2.loc[:, "Currently smokes"] = 1

    elif train_id.SmokingStatus[0] == "Ex-smoker":

        train_df2.loc[:, "Ex-smoker"] = 1

    else:

        train_df2.loc[:, "Never smoked"] = 1

        

    train_df2 = train_df2.interpolate('linear', order=2, limit_direction='both')

    train_df_Y = train_df_Y.interpolate('linear', order=2, limit_direction='both')

    train_df_Y = train_df_Y.astype('int')

    train_df_Y = train_df_Y.drop(["Weeks"], axis=1)

    

    return train_df2, train_df_Y
train_layer(0)[0]
train_layer(0)[1]
X_train = train_layer(0)[0].to_numpy()

Y_train = train_layer(0)[1].to_numpy()

sums= 0



for i in range(1, len(Patient_list)):

    a = train_layer(i)[0].to_numpy()

    X_train = np.append(X_train, a, axis=0)

    

    b = train_layer(i)[1].to_numpy()

    Y_train = np.append(Y_train, b)    
X_train.shape
Y_train.shape
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

test_df
Week = np.arange(-12, 134)

Patient_list_test = list(test_df.Patient.unique())



def test_layer (ID_N):

    test_df2   = pd.DataFrame(Week, columns = ["Weeks"])

    test_df_Y  = pd.DataFrame(Week, columns = ["Weeks"])

    test_df_Y.insert(1, 'FVC', np.nan)

    test_df2.insert(1, 'Percent', np.nan)

    test_df2.insert(2, 'Age', np.nan)

    test_df2.insert(3, 'Sex_Male', 0)

    test_df2.insert(4, 'Sex_Female', 0)

    test_df2.insert(5, 'Currently smokes', 0)

    test_df2.insert(6, 'Ex-smoker', 0)

    test_df2.insert(7, 'Never smoked', 0)



    test_id = test_df.loc[test_df.Patient == Patient_list_test[ID_N]]

    test_id = test_id.reset_index()



    for i, D in enumerate(test_id.Weeks):

        D = D + 12

        if D <= 133:

            test_df_Y.at[D, "FVC"] = test_id.FVC[i]

            test_df2.at[D, "Percent"] = test_id.Percent[i]



    test_df2.loc[:, "Age"] = test_id.Age[0]



    if test_id.Sex[0] == "Male":

        test_df2.loc[:, "Sex_Male"] = 1

    else:

        test_df2.loc[:, "Sex_Female"] = 1

    

    if test_id.SmokingStatus[0] == "Currently smokes":

        test_df2.loc[:, "Currently smokes"] = 1

    elif test_id.SmokingStatus[0] == "Ex-smoker":

        test_df2.loc[:, "Ex-smoker"] = 1

    else:

        test_df2.loc[:, "Never smoked"] = 1

        

    test_df2 = test_df2.interpolate('linear', order=2, limit_direction='both')

    test_df_Y = test_df_Y.interpolate('linear', order=2, limit_direction='both')

    test_df_Y = test_df_Y.astype('int')

    test_df_Y = test_df_Y.drop(["Weeks"], axis=1)

    

    return test_df2, test_df_Y
test_layer(0)[0]
X_test = test_layer(0)[0].to_numpy()

Y_test = test_layer(0)[1].to_numpy() #Do not use Y_test

sums= 0



for i in range(1, len(Patient_list_test)):

    a = test_layer(i)[0].to_numpy()

    X_test = np.append(X_test, a, axis=0)

    

    b = test_layer(i)[1].to_numpy()

    Y_test = np.append(Y_test, b)    
X_test.shape
Y_test.shape
import sklearn

from sklearn.linear_model import LogisticRegression

from sklearn import tree 

import lightgbm as lgb



#def FitModel(X, Y, max_depth):

#    model = LogisticRegression(max_iter=max_depth, verbose=0)

#    model.fit(X, Y)

#    return model
trainY = []

testY = []



params = {

    'metric' : 'rmse',

    'num_leaves': 100}

lgb_train = lgb.Dataset(X_train, Y_train)

model = lgb.train(params, lgb_train,)



trainY.append(model.predict(X_train))

testY.append(model.predict(X_test))
plt.figure(figsize=(18,6))



Y_train_Graph = pd.DataFrame(trainY[-1])

#Y_train = Y_train.reset_index(drop=True)

plt.plot(Y_train)

plt.plot(Y_train_Graph, label = "Predict")

plt.legend()
#DON'T USE THIS

"""

plt.figure(figsize=(18,8))



ID_NUM = 0



for i in range(5):

    Y_test_Graph = pd.DataFrame(testY[i])

    plt.plot(Y_test, linestyle = "dashed")

    plt.plot(Y_test_Graph, label = "Predict:{0}".format(i))

    plt.xlim(145*ID_NUM, 145*(ID_NUM+1))

    

plt.legend()

"""
#DON'T USE THIS

'''

import math

from scipy import stats



df_test = pd.DataFrame(testY)

Confidence = []



for i in range(df_test.shape[1]):

    data = np.array(df_test.iloc[:, i])

    Confidence1 =  df_test.iloc[:, i].mean() - (2.086*np.std(data)/math.sqrt(21))

    Confidence2 =  df_test.iloc[:, i].mean() + (2.086*np.std(data)/math.sqrt(21))

    Confidence.append(Confidence2 - Confidence1)

'''
#len(Confidence)
submission = pd.DataFrame(columns = ["Patient_Week", "FVC", "Confidence"])



WEEK = -12

D = 0



NUM = len(Patient_list_test)



for j in range(146):

    

    for i in range(NUM):

        submission.loc[j*NUM + i,"Patient_Week"] = Patient_list_test[i] +"_" + str(WEEK)

        submission.loc[j*NUM + i,"FVC"] = Y_test[i*146 + j]

        submission.loc[j*NUM + i,"Confidence"] = 285



    WEEK = WEEK + 1
submission.to_csv('submission.csv', index=False)
submission.head(40)