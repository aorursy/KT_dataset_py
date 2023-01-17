# Loading necessary packages 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

%matplotlib inline
import seaborn as sns
sns.set()

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier, plot_importance
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
# loading dataset
dataset = pd.read_csv("../input/xAPI-Edu-Data.csv")
# A summary of the dataset
dataset.info()
# breif description of the numerical valued feature
dataset.describe()
plt.subplots(figsize=(20, 8))
dataset["raisedhands"].value_counts().sort_index().plot.bar()
plt.title("No. of times vs no. of students raised their hands on particular time", fontsize=18)
plt.xlabel("No. of times, student raised their hands", fontsize=14)
plt.ylabel("No. of student, on particular times", fontsize=14)
plt.show()
plt.subplots(figsize=(20, 8))
dataset["VisITedResources"].value_counts().sort_index().plot.bar()
plt.xlabel("No. of times, student visted resource", fontsize=14)
plt.ylabel("No. of student, on particular visit", fontsize=14)
plt.show()
dataset.groupby("gender").count()
def gen_bar(feature, size):
    highest = dataset[dataset["Class"]=="H"][feature].value_counts()
    medium = dataset[dataset["Class"]=="M"][feature].value_counts()
    low = dataset[dataset["Class"]=="L"][feature].value_counts()
    
    df = pd.DataFrame([highest, medium, low])
    df.index = ["highest","medium", "low"]
    df.plot(kind='bar',stacked=True, figsize=(size[0], size[1]))
gen_bar("gender",[6,5])
# lets map the gender
gender_map = {"F" : 0, "M" : 1}
dataset["gender"] = dataset["gender"].map(gender_map)
dataset["gender"].value_counts().sort_index().plot.bar()
dataset["NationalITy"].describe()
plt.subplots(figsize=(15, 5))
dataset["NationalITy"].value_counts().sort_index().plot.bar()
dataset["NationalITy"] = dataset["NationalITy"].replace(["Jordan","KW"], "0")
dataset["NationalITy"] = dataset["NationalITy"].replace(["Iraq","Palestine"], "1")
dataset["NationalITy"] = dataset["NationalITy"].replace(["Tunis","lebanon", "SaudiArabia"], "2")
dataset["NationalITy"] = dataset["NationalITy"].replace(["Syria","Egypt","Iran","Morocco","USA","venzuela","Lybia"], "3")

dataset["NationalITy"] = dataset["NationalITy"].astype(int)
plt.subplots(figsize=(5, 5))
dataset["NationalITy"].value_counts().sort_index().plot.bar()
del dataset["PlaceofBirth"]
dataset["StageID"].value_counts().sort_index().plot.bar()
stage_map = {"HighSchool" : 0, "MiddleSchool" : 1, "lowerlevel": 2}
dataset["StageID"] = dataset["StageID"].map(stage_map)
gen_bar("GradeID",[8,8])
dataset["GradeID"] = dataset["GradeID"].replace(["G-02","G-08","G-07"], "0")
dataset["GradeID"] = dataset["GradeID"].replace(["G-04","G-06"], "1")
dataset["GradeID"] = dataset["GradeID"].replace(["G-05","G-11", "G-12","G-09","G-10"], "2")

dataset["GradeID"] = dataset["GradeID"].astype(int)
dataset.groupby("SectionID").count()
section_map = {"A":0, "B":1, "C":2}
dataset["SectionID"] = dataset["SectionID"].map(section_map)
plt.subplots(figsize=(15, 5))
dataset["Topic"].value_counts().sort_index().plot.bar()
gen_bar("Topic", [8,10])
pd.crosstab(dataset["Class"],dataset["Topic"])
topic_map = {"IT":0, "Arabic":1, "French":2, "English":3, "Biology":4, "Science":5, "Chemistry":6, "Quran":7, "Geology":8, "History":9,"Math":9,"Spanish":9}
dataset["Topic"] = dataset["Topic"].map(topic_map)
dataset.groupby("Topic").count()
facet = sns.FacetGrid(dataset, hue="Class",aspect=4)
facet.map(sns.kdeplot,"Topic",shade= True)
facet.set(xlim=(0, dataset["Topic"].max()))
facet.add_legend()

plt.show()
dataset.groupby("Semester").count()
pd.crosstab(dataset["Class"], dataset["Semester"])
semester_map = {"F":0, "S":1}
dataset["Semester"] = dataset["Semester"].map(semester_map)
dataset["Relation"].value_counts().sort_index().plot.bar()
relation_map = {"Father":0, "Mum":1}
dataset["Relation"] = dataset["Relation"].map(relation_map)
dataset["ParentschoolSatisfaction"].nunique()
parent_ss_map = {"Bad": 0, "Good":1}
dataset["ParentschoolSatisfaction"] = dataset["ParentschoolSatisfaction"].map(parent_ss_map)
dataset.groupby("ParentschoolSatisfaction").count()
gen_bar("ParentschoolSatisfaction", [5,5])
dataset.groupby("ParentAnsweringSurvey").count()
parent_a_s_map = {"No":0, "Yes":1}
dataset["ParentAnsweringSurvey"] = dataset["ParentAnsweringSurvey"].map(parent_a_s_map)
dataset["ParentAnsweringSurvey"].value_counts().sort_index().plot.bar()
dataset.groupby("StudentAbsenceDays").count()
student_absn_day_map = {"Above-7":0, "Under-7":1} 
dataset["StudentAbsenceDays"] = dataset["StudentAbsenceDays"].map(student_absn_day_map)
dataset.groupby("StudentAbsenceDays").count()
dataset.head()
dataset.groupby("Class").count()
class_map = {"H":0, "M":1, "L":2}
dataset["Class"] = dataset["Class"].map(class_map)
dataset.groupby("Class").count()
dataset.corr()
fig, ax = plt.subplots(figsize = (15, 10))
ax = sns.heatmap(dataset.corr(), annot = True, fmt = ".1f", cmap = "RdYlBu")
X = dataset.iloc[:,0:15]
y = dataset["Class"]

features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size = .20, random_state=0)

# model build with SVM.SVC classifier

clf = SVC(gamma='auto', kernel = 'linear')
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy_score(pred, labels_test)
# Random Forest Classifier with 20 subtrees

clf = RandomForestClassifier(n_estimators = 220, random_state=10)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
rfc_pred = pred
accuracy_score(pred, labels_test)
# Logistic regression
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy_score(pred, labels_test)
# Multi-layer Perceptron classifier with (10,10,10) hidden layers

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,10,10), random_state=1)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy_score(pred, labels_test)
clf = XGBClassifier(max_depth=5, learning_rate=0.2, n_estimators=20, seed=10)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
xgb_pred = pred
accuracy_score(pred, labels_test)
fig, ax = plt.subplots(figsize=(12,6))
plot_importance(clf, ax = ax)
# Random Forest Classifier confustion Matrix result
confusion_matrix(labels_test, rfc_pred, labels=[1, 0]) 
# XGBoost Classifier confusion matric result
confusion_matrix(labels_test, xgb_pred, labels=[1, 0]) 