# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json
import warnings
warnings.filterwarnings("ignore")


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
## -- Training Sentiment Metedata -- ##
json_path = "../input/train_sentiment"
json_list = os.listdir(json_path)

jdict_train = {}
for js in json_list:
    with open(os.path.join(json_path, js), encoding="utf8") as json_file:
        json_text = json.load(json_file)
        jdict_train[js[0:-5]] = json_text["documentSentiment"]["magnitude"]
## -- Testing Sentiment Metedata -- ##
json_path = "../input/test_sentiment"
json_list = os.listdir(json_path)

jdict_test = {}
for js in json_list:
    with open(os.path.join(json_path, js), encoding="utf8") as json_file:
        json_text = json.load(json_file)
        jdict_test[js[0:-5]] = json_text["documentSentiment"]["magnitude"]

print(len(jdict_train))
print(len(jdict_test))

df_train_data = pd.read_csv("../input/train/train.csv", header=0, index_col="PetID")
df_test_data = pd.read_csv("../input/test/test.csv", header=0, index_col="PetID")

df_sent_train = pd.DataFrame.from_dict(jdict_train, orient="index", columns=["Magnitude"])
df_sent_test = pd.DataFrame.from_dict(jdict_test, orient="index", columns=["Magnitude"])

joined_train = df_train_data.merge(df_sent_train, how="left", left_index=True, right_index=True)
joined_train["PredictionType"] = "Training"
joined_train["Magnitude"] = joined_train["Magnitude"].fillna(value=joined_train["Magnitude"].mean())
                                                             
joined_test = df_test_data.merge(df_sent_test, how="left", left_index=True, right_index=True)
joined_test["PredictionType"] = "Testing"
joined_test["AdoptionSpeed"] = 100
joined_test["Magnitude"] = joined_test["Magnitude"].fillna(value=joined_test["Magnitude"].mean())

full_df =  joined_train.append(joined_test)

print(full_df.info())
data_type_dict = {
        "category":["Type", # 1 = Dog, 2 = Cat
                    "Breed1", # Primary breed of pet (Refer to BreedLabels dictionary)
                    "Breed2", # Secondary, if mixed
                    "Gender", # 1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets
                    "Color1", # Color 1 of pet (Refer to ColorLabels dictionary)
                    "Color2", 
                    "Color3", 
                    "MaturitySize", # Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)
                    "FurLength", # Fur length (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)
                    "Vaccinated", # Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)
                    "Dewormed", # Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)
                    "Sterilized", # Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)
                    "Health", # Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)
                    "State" # State location in Malaysia (Refer to StateLabels dictionary)								
                    ],
        "float":["Age", # Months
                 "Quantity", # Number of pets represented in profile
                 "Fee", # Adoption fee (0=Free)
                 "VideoAmt", # Total uploaded videos for this pet
                 "PhotoAmt", # Total uploaded photos for this pet
                 "Magnitude", # From the Sentiment metadata
                 ],
        "int8":["AdoptionSpeed"], # Categorical speed of adoption. Lower is faster. This is the value to predict.]
        "object":["PredictionType"] # Custom indeifyer of testing/training set"
                 }

output_var = "AdoptionSpeed"


# df_train = pd.DataFrame()
df_learn = pd.DataFrame()
for typ,col in data_type_dict.items():
    for c in col:
        df_learn[c] = full_df[c].astype(typ)

print(df_learn.info())
df_learn = pd.get_dummies(df_learn)

X_train_whole = df_learn[df_learn["PredictionType_Training"] == 1].drop(output_var, axis=1)
y_train_whole = df_learn[df_learn["PredictionType_Training"] == 1][output_var]

from sklearn.model_selection import train_test_split
rnd = 0
X_train, X_test, y_train, y_test = train_test_split(X_train_whole, y_train_whole, 
                                                    test_size=0.20, 
                                                    random_state=rnd).copy()
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


models = {"Random Forest":RandomForestClassifier(random_state=rnd),
          "K-neighbors":KNeighborsClassifier(),
          "Linear SVC":LinearSVC(random_state=rnd)
         }

for desc, mod in models.items():
    print(desc)
    mod.fit(X_train, y_train)
    print(mod.score(X_test, y_test))
mod = RandomForestClassifier(n_estimators=120, random_state=rnd)
mod.fit(X_train_whole, y_train_whole)
pred = mod.predict(df_learn[df_learn["PredictionType_Testing"] == 1].drop(output_var, axis=1))

df_test_data["AdoptionSpeed"] = pred
submission = df_test_data["AdoptionSpeed"]
print(submission.head(10).to_string())
submission.to_csv("submission.csv", index=True, index_label="PetID", header=["AdoptionSpeed"])



