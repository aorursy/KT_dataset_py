# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
os.chdir("../input")

train = pd.read_csv("train/train.csv")

test = pd.read_csv("test/test.csv")
train_plt = train.drop(["Name","RescuerID","Description","PetID"],axis=1)

test_plt = test.drop(["Name","RescuerID","Description","PetID"],axis=1)

fig = plt.figure(figsize=(20,20))

x = 1

for col in test_plt.columns:

    sns.countplot(x=col,data=train,ax=fig.add_subplot(9,6,x)).set_title("train_"+col)

    x += 1

    sns.countplot(x=col,data=test,ax=fig.add_subplot(9,6,x)).set_title("test_"+col)

    x += 1

    fig.tight_layout()
# add "image_num" column to train.

import glob



train_img_list = glob.glob("./train_images/*.jpg")

test_img_list = glob.glob("./test_images/*.jpg")
# get PetID from filename(./train_images/PetID-num.jpg)

train_img_list = [fname.split("/")[2].split("-")[0] for fname in train_img_list]

test_img_list = [fname.split("/")[2].split("-")[0] for fname in test_img_list]
import collections



train_img_count = collections.Counter(train_img_list)

train["img_count"] = 0

for key,value in train_img_count.items():

    train.loc[train.PetID == key,"img_count"] = value

    

test_img_count = collections.Counter(test_img_list)

test["img_count"] = 0

for key,value in test_img_count.items():

    test.loc[test.PetID == key,"img_count"] = value
#make lists from excepted unique data (e.g. ID) to numerical data and categorical data

numerical_cols=["Age", "Quantity", "Fee", "VideoAmt", "PhotoAmt","img_count"]

categorical_cols1=["Type", "Gender", "MaturitySize", "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health"]

categorical_cols2=["Type","Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3", "MaturitySize", "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health", "State"]
train_dummy = train[numerical_cols + categorical_cols1]

test_dummy = test[numerical_cols + categorical_cols1]

target = train["AdoptionSpeed"]
from pandas.api.types import CategoricalDtype



breed_df = pd.read_csv("breed_labels.csv")

color_df = pd.read_csv("color_labels.csv")

state_df = pd.read_csv("state_labels.csv")



breed_cat = CategoricalDtype(categories = breed_df["BreedID"].unique(), ordered=False)

color_cat = CategoricalDtype(categories = color_df["ColorID"].unique(), ordered=False)

state_cat = CategoricalDtype(categories = state_df["StateID"].unique(), ordered=False)



train_dummy["Breed1"] = train["Breed1"].astype(breed_cat)

train_dummy["Breed2"] = train["Breed2"].astype(breed_cat)

train_dummy["Color1"] = train["Color1"].astype(color_cat)

train_dummy["Color2"] = train["Color2"].astype(color_cat)

train_dummy["Color3"] = train["Color3"].astype(color_cat)

train_dummy["State"] = train["State"].astype(state_cat)



test_dummy["Breed1"] = test["Breed1"].astype(breed_cat)

test_dummy["Breed2"] = test["Breed2"].astype(breed_cat)

test_dummy["Color1"] = test["Color1"].astype(color_cat)

test_dummy["Color2"] = test["Color2"].astype(color_cat)

test_dummy["Color3"] = test["Color3"].astype(color_cat)

test_dummy["State"] = test["State"].astype(state_cat)
train_dummy = pd.get_dummies(train_dummy, columns = categorical_cols2, drop_first = True)

test_dummy = pd.get_dummies(test_dummy, columns = categorical_cols2, drop_first = True)
from sklearn import datasets

from sklearn.ensemble import RandomForestClassifier as RFC

from sklearn.model_selection import train_test_split, GridSearchCV
Tr_train, Test_train, Tr_target, Test_target = train_test_split(train_dummy, target, test_size=0.2)
search_params = {

    'n_estimators'      : [100],

    'criterion'         : ['gini', 'entropy'],

    'max_features'      : ["auto", 3, 20],

    'random_state'      : [2525],

    'n_jobs'            : [1],

    'min_samples_split' : [3, 10, 20, 50, 100],

    'max_depth'         : [3, 10, 20, 50, 100],

    'bootstrap'         : [False],#True

    'oob_score'         : [False],#True

}

gs = GridSearchCV(RFC(),search_params, cv=2, verbose=True, n_jobs=-1)

gs.fit(Tr_train, Tr_target)

 

print(gs.best_estimator_)
print(f"acc: {gs.score(Test_train, Test_target)}")
# submit_data

submit = pd.DataFrame()

submit["PetID"] = test["PetID"]

submit["AdoptionSpeed"] = gs.predict(test_dummy)

submit.to_csv("../working/submission.csv",index=False)