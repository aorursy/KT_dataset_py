#importing required libraries to read and clean raw data
import pandas as pd
import numpy as np
data = pd.read_csv("../input/StudentsPerformance.csv")
data.head()
#getting basic info about the data frame
data.info()
data.isna().sum() # returns total no. of rows which contain null or missing values
data.columns = ["gender", "race","parent_edu", "lunch", "test_prep", "score_math", "score_reading", "score_writing"] #passing a list of string to rename columns
data.columns #re-checking work!
data["race"] = data["race"].str.replace("group", "")
data.head()
#introducing categorical variables "P" and "F" to indicate pass/fail status
data["math_status"] = np.where(data["score_math"] >=40, "P", "F")
data["reading_status"]= np.where(data["score_reading"] >=40, "P", "F")
data["writing_status"] = np.where(data["score_writing"] >=40, "P", "F")
data.head(3)

#importing required libraries for data visualisation

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = 8,6
sns.countplot(x = data["gender"]).set_title("Sex ratio of students")
plt.show()
total = data["gender"].shape[0]
female = pd.value_counts(data["gender"] == "female")
female_percent = (female/total)*100
male_percent = 1 - female_percent
print(female_percent)

# 51.7% students are female and 48.2% students are male
plt.rcParams["figure.figsize"] = 8,6
sns.countplot(x = data["race"]).set_title("Race/ethnicity distribution of students")
plt.show()
plt.rcParams["figure.figsize"] = 12,7
sns.countplot(x = data["parent_edu"]).set_title("Parent's education level")
plt.show()

plt.rcParams["figure.figsize"] = 8,8
sns.countplot(x = data["gender"], hue = data["test_prep"]).set_title("Preparation of students")
plt.show()
plt.rcParams["figure.figsize"] = 8,8
sns.countplot(x = data["gender"], hue = data["lunch"]).set_title("Lunch types")
plt.show()
fig, axarr = plt.subplots(2, 2)
ax1, ax2, ax3, ax4 = axarr[0, 0], axarr[0, 1], axarr[1, 0], axarr[1, 1]
fig.set_figheight(15)
fig.set_figwidth(25)

sns.pointplot(y=data['score_math'], x=data["race"], ax=ax1)
sns.pointplot(y=data['score_math'], x=data['gender'], ax=ax2)
sns.pointplot(y=data['score_math'], x=data["parent_edu"],ax=ax3)
sns.pointplot(y=data['score_math'], x=data['gender'], hue= data["test_prep"],ax=ax4)
plt.show()
fig, axarr = plt.subplots(2, 2)
ax1, ax2, ax3, ax4 = axarr[0, 0], axarr[0, 1], axarr[1, 0], axarr[1, 1]
fig.set_figheight(15)
fig.set_figwidth(25)

sns.pointplot(y=data['score_reading'], x=data["race"], ax=ax1)
sns.pointplot(y=data['score_reading'], x=data['gender'], ax=ax2)
sns.pointplot(y=data['score_reading'], x=data["parent_edu"],ax=ax3)
sns.pointplot(y=data['score_reading'], x=data['gender'], hue= data["test_prep"],ax=ax4)
plt.show()
fig, axarr = plt.subplots(2, 2)
ax1, ax2, ax3, ax4 = axarr[0, 0], axarr[0, 1], axarr[1, 0], axarr[1, 1]
fig.set_figheight(15)
fig.set_figwidth(25)

sns.pointplot(y=data['score_writing'], x=data["race"], ax=ax1)
sns.pointplot(y=data['score_writing'], x=data['gender'], ax=ax2)
sns.pointplot(y=data['score_writing'], x=data["parent_edu"],ax=ax3)
sns.pointplot(y=data['score_writing'], x=data['gender'], hue= data["test_prep"],ax=ax4)
plt.show()
a = pd.value_counts(data["lunch"])
print(a)

#35.5% students qualified for the lunch fees waiver
plt.rcParams["figure.figsize"] = 15,10
sns.countplot(x = data["parent_edu"], hue = data["lunch"]).set_title("Parents education levels qualifying for lunch wiaver")
plt.show()
