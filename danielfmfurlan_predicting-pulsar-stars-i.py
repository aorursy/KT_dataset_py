# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path = "/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv"

stars = pd.read_csv(path)

results = []

print("size of our data : ", len(stars))

stars.head()
print("Checking if there is missing values :\n",stars.isnull().sum())
stars.columns

stars.rename(columns={" Excess kurtosis of the DM-SNR curve": " Excess kurtosis of the DM SNR curve"," Skewness of the DM-SNR curve":"Skewness of the DM SNR curve", " Mean of the DM-SNR curve" : " Mean of the DM SNR curve", " Standard deviation of the DM-SNR curve" : " Standard deviation of the DM SNR curve" },inplace=True)

stars.columns
cor = stars.corr()
import seaborn as sns; sns.set()

plt.figure(figsize=(18,10))





ax = sns.heatmap(

    cor, 

    center=0,

    vmin = -1, vmax = 1.0,

    linewidth=.9,

    cmap =  sns.color_palette("RdBu_r", 7),#cmap="YlGnBu",

    annot=True,

    square=True,



)
mask = np.zeros_like(cor)

mask[np.triu_indices_from(mask)] = True



with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(18, 10))

    ax = sns.heatmap(cor, center = 0, linewidth = 0.9, vmin = -1, vmax = 1, 

    cmap =  sns.color_palette("RdBu_r", 7),annot = True, mask=mask, square=True)

    
corr_m = cor.abs()

sol = (corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(np.bool))

                 .stack()

                 .sort_values(ascending=False))

sol[:3]
from heapq import nlargest

from operator import itemgetter

import itertools

from itertools import combinations

from scipy import stats

all_cor = []

all_cord = {}

for one,two in itertools.combinations(stars.columns,2):



    v,_ = stats.pearsonr(stars[one],stars[two])

    all_cor.append(f"Correlation btw {one} and {two} is: {v:.4f}")

    all_cord[one+"-"+two] = round(v,4)



all_cor



m = dict(sorted(all_cord.items(), key = itemgetter(1), reverse = True)[:3])

print("The 3 strongest correlations are : ",m)


import matplotlib.pyplot as plt



i = 0

plt.figure()

f, ax = plt.subplots(1, 3, figsize=(21, 7), sharex=True)

#for first,sec in itertools.combinations(stars.columns.drop(stars.columns[len(stars.columns)-1]),2):

for key,val in m.items():

    #print(first,sec)

    pair = key.split("-")

    first = pair[0]

    sec = pair[1]

    

    sns.scatterplot(x=first, y=sec, data=stars, hue="target_class", ax = ax[i])

    if i == 2:

        break

    i += 1

    
fr = 0.1

vsize = int(len(stars)*fr)



train = stars[:-2*vsize]

valid = stars[-2*vsize:-vsize]

test = stars[:-vsize]



for each in [train,valid,test]:

    print(f"Percentage of target values : {stars.target_class.mean():.4f}")
import lightgbm as lgb

from sklearn import metrics

val_pred = []

ground = []

def training(feat_cols):

    plt.figure()

    global val_pred, ground

    evals_result = {}

    

    dtrain = lgb.Dataset(data=train[feat_cols], label=train["target_class"])

    dvalid = lgb.Dataset(data=valid[feat_cols], label=valid["target_class"])

    dtest = lgb.Dataset(data=test[feat_cols], label=test["target_class"])



    param = {"num_leaves" : 64, "objectives":"binary"}

    param["metric"] = "auc"



    num_round = 500

    bst = lgb.train(param,dtrain,num_round,valid_sets=[dvalid],evals_result = evals_result, early_stopping_rounds = 10)

    

    #lgb.plot_metric(evals_result, metric="auc", figsize=(7,7))

    lgb.plot_importance(bst, max_num_features=10,figsize=(10,10))

    

    ypred = bst.predict(test[feat_cols])

    score = metrics.roc_auc_score(test["target_class"], ypred)



    val_pred = ypred

    ground = test["target_class"]

    

    print(f"our score is: {score:.4f}")

    return score, dvalid


features = []

for key,val in m.items():

    feat = key.split("-")

    for each2 in feat:

        if each2 not in features:

            features.append(each2)

#features

res = {"baseline":"","selected features":""}



res["selected features"],_ = (training(features))



## With all columns:

feat_cols = stars.columns.drop("target_class")

res["baseline"],_ = (training(feat_cols))



res



diferr = pd.DataFrame(columns=["Prediction", "Ground_Truth"])

diferr["Ground_Truth"] = ground

diferr["Prediction"] = val_pred

diferr

print("Predictions for label = 0. Not pulsar stars\n",diferr.loc[diferr["Ground_Truth"]==0])

print("Predictions for label = 1. Pulsar stars\n",diferr.loc[diferr["Ground_Truth"]==1])
from sklearn.model_selection import train_test_split

from sklearn.metrics import (roc_curve, auc, accuracy_score)

from sklearn.model_selection import GridSearchCV



plt.figure(figsize=(10,10))

valAcc = accuracy_score(ground, np.round(val_pred))

fprVal, tprVal, thresholdsVal = roc_curve(ground, val_pred)

valAUC =  auc(fprVal, tprVal)

print("Our threscholds : {}. Type : {}. Lenght : {}".format(thresholdsVal, type(thresholdsVal), len(thresholdsVal)))

print("valAUC : {} and valAcc : {}".format(valAUC, valAcc))

#Plot ROC curve from tpr and fpr.

plt.plot(fprVal, tprVal, label="Validation")

plt.legend()

plt.ylabel('True positive rate.')

plt.xlabel('False positive rate')

plt.title("ROC curve for validation")

plt.show()
import csv

csvfile = "/kaggle/working/results.csv"



with open(csvfile, "w") as output:

    writer = csv.writer(output, lineterminator='\n')

    for val in results:

        writer.writerow([val])  
