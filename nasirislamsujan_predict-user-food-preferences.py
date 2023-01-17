# loading necessary packages



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE, ADASYN

from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler

from sklearn.model_selection import train_test_split, cross_validate

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_validate

from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, recall_score, f1_score, 

                             precision_score, make_scorer)
dataset = pd.read_csv("../input/data.csv")
dataset.head()
# shape of the dataset

# there are 323138 observations and 8 features 



dataset.shape
dataset.describe()
# top 10 most food ordered users 

# by default value_counts() method sort the result in descending order

dataset.user_id.value_counts().head(10)
# number of unique items

len(dataset.item_id.unique())
# number of times Each item ordered by user

dataset.item_id.value_counts().head(15)
items = dataset.item_id.values



# concatenate all the items into a large string 

all_items = ",".join(items) 



# Create and generate a word cloud image (1500 x 800 px):

wordcloud = WordCloud(width=1500, height=800).generate(all_items)
plt.figure(figsize=(20, 8))

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# mapping data for better readability

day_map = {1: "Saturday", 2: "Sunday", 3:"Monday", 4:"Tuesday", 5: "Wednesday", 6: "Thursday", 7:"Friday"}

dataset.dow = dataset.dow.map(day_map)
# total items ordered by users each day of week

dataset.dow.value_counts()
dataset.dow.value_counts().plot.bar()
plt.figure(figsize=(10, 5))

sns.color_palette("Set2")

sns.countplot(x = "dow", hue="item_count", data = dataset)

plt.xlabel("Day of Week", fontsize=14)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.title("Number of Items ordered in a particular day", fontsize=16)

plt.show()
dataset[dataset.item_count>3].head()
# only 48 times users ordered more than 3 items at a time

len(dataset[dataset.item_count>3])
dataset["dow"][dataset.item_count>3].value_counts()
dataset.hod.value_counts()
dataset.hod.value_counts().plot.bar()
facet = sns.FacetGrid(dataset, hue ="item_count", aspect = 4)

facet.map(sns.kdeplot,"hod", shade = True)

facet.set(xlim = (0, dataset["hod"].max()))

facet.add_legend()



plt.show()
# there are 15131 unique categories are avialble 

len(dataset.category_id.unique())
# Number of time each category item ordered by users

dataset.category_id.value_counts().head(10)
categories = dataset.category_id.values



# concatenate all the categories into a large string 

all_categories = ",".join(map(str, categories))



# Create and generate a word cloud image (1500 x 800 px):

wordcloud = WordCloud(width=1500, height=800).generate(all_categories)
plt.figure(figsize=(20, 8))

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# there are 47 types of cusines are available

len(dataset["cusine_id"].unique())
# user's ordered from each cusines

plt.figure(figsize=(16, 6))

dataset["cusine_id"].value_counts().plot.bar()
pd.set_option("max_column", None)

# which category belongs to which cusine 

pd.crosstab(dataset["category_id"], dataset["cusine_id"]).head(10)
# most popular cusines

cusines = dataset.cusine_id.values



# concatenate all the cusines into a large string 

all_cusines = ",".join(map(str, cusines))



# Create and generate a word cloud image (1500 x 800 px):

wordcloud = WordCloud(width=1500, height=800).generate(all_cusines)
plt.figure(figsize=(20, 8))

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# 3357 unique resturants

len(dataset["restaurant_id"].unique())
# which resturants serves which cusines

pd.crosstab(dataset["restaurant_id"], dataset["cusine_id"]).head(5)
# total 871 restaurant serves `2e0c31a5-850` this cusine

len(dataset["restaurant_id"][dataset["cusine_id"]=="2e0c31a5-850"].unique())
# top 10 restaurants

dataset["restaurant_id"].value_counts().head(10)
dataset["restaurant_id"].value_counts().head(10).plot.bar()
len(dataset[dataset.item_count==5]), len(dataset[dataset.item_count==4]), len(dataset[dataset.item_count==3]), \

len(dataset[dataset.item_count==2]), len(dataset[dataset.item_count==1])
# 5 new feature extraction

dataset["single_meal"] = (dataset.item_count==1).astype(int)

dataset["couple_meal"] = (dataset.item_count==2).astype(int)

dataset["small_treat"] = (dataset.item_count==3).astype(int)

dataset["medium_treat"] = (dataset.item_count==4).astype(int)

dataset["large_treat"] = (dataset.item_count==5).astype(int)
# handling missing values

dataset.fillna("0", inplace=True)
day_map = {"Saturday" : 1, "Sunday" : 2,"Monday": 3, "Tuesday": 4, "Wednesday" :5 , "Thursday" : 6, "Friday": 7}

dataset.dow = dataset.dow.map(day_map)



# label encoding

encoder = LabelEncoder()

dataset["item_id"] = encoder.fit_transform(dataset["item_id"])

dataset["category_id"] = encoder.fit_transform(dataset["category_id"])

dataset["cusine_id"] = encoder.fit_transform(dataset["cusine_id"])

dataset["restaurant_id"] = encoder.fit_transform(dataset["restaurant_id"])
dataset.head()
# imbalance problem

print(dataset["item_count"].value_counts())

dataset["item_count"].value_counts().plot.bar()
X = dataset.drop(["user_id", "item_count"], axis=1)

y = dataset["item_count"]



smote = SMOTE(random_state=0)



X_sm, y_sm = smote.fit_sample(X, y)
# each class has now equal instances 

pd.Series(y_sm).value_counts()
ada = ADASYN(random_state=0)

X_ada, y_ada = ada.fit_sample(X, y)
# ADASYN does not create equal synthetic instance for all classes

pd.Series(y_ada).value_counts()
cc = ClusterCentroids(random_state=0)

X_cc, y_cc = cc.fit_sample(X, y)
pd.Series(y_cc).value_counts()
rus = RandomUnderSampler()

X_rus, y_rus = rus.fit_sample(X, y)
pd.Series(y_rus).value_counts()
X_train, X_test, y_train, y_test = train_test_split(X_ada, y_ada, test_size=0.2, random_state=42)
clf_rf = RandomForestClassifier()

pred = clf_rf.fit(X_train, y_train).predict(X_test)

acc = accuracy_score(pred, y_test)

acc
cm = confusion_matrix(pred, y_test)

cm
sns.heatmap(cm, annot= True, fmt=".0f")
print(classification_report(pred, y_test))
# 5-fold Cross Validation



clf = RandomForestClassifier()



scoring = {'accuracy': make_scorer(accuracy_score), 

           'precision': make_scorer(precision_score, average='macro'),

           'recall': make_scorer(recall_score, average='macro'),

           'f1': make_scorer(f1_score, average='macro'),

          }



scores = cross_validate(clf, X, y, cv = 5, scoring = scoring, return_train_score=True)
scores
print("Avg. Accuracy ", scores["test_accuracy"].mean())

print("Avg. Precision ", scores["test_precision"].mean())

print("Avg. Recall ", scores["test_recall"].mean())

print("Avg. F1 ", scores["test_f1"].mean())