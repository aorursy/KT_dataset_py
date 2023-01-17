import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



import csv

import matplotlib.pyplot as plt

import seaborn as sns

"""

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

"""

import statistics
# defining directory paths

train_dir = "../input/train.csv"

#test_dir = "../input/test.csv"
df = pd.read_csv(train_dir)

#checking for missing and na values

print("Total number of instance : ",len(df))

df.isna().sum()
df.fillna(-1, inplace = True) #filling in missing values with -1

df["Cabin"].unique()
df.drop(["Cabin"], axis = 1, inplace = True)

df.drop(["Ticket"], axis = 1, inplace = True)

df.info()

df.head(3)
# checking class distribution

df["Survived"].value_counts().plot(kind = "bar")
f,ax = plt.subplots(figsize=(6, 5))

sns.heatmap(df.corr(), annot=True, cmap = "Blues", linewidths=.5, fmt= '.2f',ax = ax)

plt.show()
df_survived = df[df["Survived"]==1]

df_notsurvived = df[df["Survived"]==0]

gb_pclass_surv = df_survived.groupby("Pclass")["Survived"].sum()

gb_pclass_notsurv = df_notsurvived.groupby("Pclass")["Survived"].count()



fig = plt.figure(figsize = (10,4))

f1 = fig.add_subplot(1, 2, 1)

f1.set_ylim([0,400])

f2 = fig.add_subplot(1,2,2)

f2.set_ylim([0,400])

gb_pclass_surv.plot(kind= "bar", title = "Survived", ax = f1)

gb_pclass_notsurv.plot(kind= "bar", title = "Not Survived", ax = f2)
df.drop("PassengerId", axis = 1, inplace = True)
print("SibSp unqiue value counts :\n" + str(df["SibSp"].value_counts()))



fig = plt.figure(figsize = (15,5))

f1 = fig.add_subplot(1, 3, 1)

f1.set_ylim([0,700])

f2 = fig.add_subplot(1,3,2)

f2.set_ylim([0,700])

f3 = fig.add_subplot(1,3, 3)

f3.set_ylim([0,700])

df["SibSp"].value_counts().plot(kind= "bar", title = "(SibSp) Total", ax = f1)

df_survived["SibSp"].value_counts().plot(kind= "bar", title = "(SibSp) Survived", ax = f2)

df_notsurvived["SibSp"].value_counts().plot(kind= "bar", title =  "(SibSp) Not Survived", ax = f3)

plt.show()
print("Parch unique value counts : \n" + str(df["Parch"].value_counts()))



fig = plt.figure(figsize = (15,5))

f1 = fig.add_subplot(1, 3, 1)

f1.set_ylim([0,700])

f2 = fig.add_subplot(1,3,2)

f2.set_ylim([0,700])

f3 = fig.add_subplot(1,3, 3)

f3.set_ylim([0,700])

df["Parch"].value_counts().plot(kind= "bar", title = "(Parch) Total", ax = f1)

df_survived["Parch"].value_counts().plot(kind= "bar", title = "(Parch) Survived", ax = f2)

df_notsurvived["Parch"].value_counts().plot(kind= "bar", title =  "(Parch) Not Survived", ax = f3)

plt.show()
df["Sex"].replace("male", 0, inplace = True)

df["Sex"].replace("female", 1, inplace = True)



df["Embarked"].replace(["S","C","Q"],[0,1,2], inplace = True)
df["n_fam_mem"] = df["SibSp"] + df["Parch"]

df_survived["n_fam_mem"] = df_survived["SibSp"] + df_survived["Parch"]

df_notsurvived["n_fam_mem"] = df_notsurvived["SibSp"] + df_notsurvived["Parch"]



fig = plt.figure(figsize = (15,5))

f1 = fig.add_subplot(1, 3, 1)

f1.set_ylim([0,600])

f2 = fig.add_subplot(1,3,2)

f2.set_ylim([0,600])

f3 = fig.add_subplot(1,3, 3)

f3.set_ylim([0,600])



df["n_fam_mem"].value_counts().plot(kind = "bar", title = "all", ax = f1)

df_survived["n_fam_mem"].value_counts().plot(kind = "bar", title = "Survived", ax = f2)

df_notsurvived["n_fam_mem"].value_counts().plot(kind = "bar", ax = f3, title = "Not Survived")
def create_family_ranges(df):

    familysize = []

    for members in df["n_fam_mem"]:

        if members == 0:

            familysize.append(0)

        elif members > 0 and members <=4:

            familysize.append(1)

        elif members > 4:

            familysize.append(2)

    return familysize



famsize = create_family_ranges(df)

df["familysize"] = famsize

df["Age"].where(df["Age"]!=-1).mean()
def age_to_int(df):

    agelist = df["Age"].values.tolist()

    for i in range(len(agelist)):

        if agelist[i] < 18 and agelist[i] >= 0:

            agelist[i] = 0

        elif agelist[i] >= 18 and agelist[i] < 60:

            agelist[i] = 1

        elif agelist[i]>=60 and agelist[i]<200:

            agelist[i] = 2

        else:

            agelist[i] = -1

    ageint = pd.DataFrame(agelist)

    return ageint
ageint = age_to_int(df)

df["Ageint"] = ageint

df.drop("Age", axis = 1, inplace = True)

df["actual_fare"] = df["Fare"]/(df["n_fam_mem"]+1)



df["actual_fare"].plot()

df["actual_fare"].describe()
def conv_fare_ranges(df): 

    fare_ranges = []

    for fare in df.actual_fare:

        if fare < 7:

            fare_ranges.append(0)

        elif fare >=7 and fare < 14:

            fare_ranges.append(1)

        elif fare >=14 and fare < 30:

            fare_ranges.append(2)

        elif fare >=30 and fare < 50:

            fare_ranges.append(3)

        elif fare >=50:

            fare_ranges.append(4)

    return fare_ranges

        

fare_ranges = conv_fare_ranges(df)

df["fare_ranges"] = fare_ranges

df_nonsurv_fare = df[df["Survived"]==0]

df_surv_fare = df[df["Survived"]==1]



fig = plt.figure(figsize = (15,5))

f1 = fig.add_subplot(1, 3, 1)

f1.set_ylim([0,500])

f2 = fig.add_subplot(1,3,2)

f2.set_ylim([0,500])

f3 = fig.add_subplot(1,3, 3)

f3.set_ylim([0,500])



df["fare_ranges"].value_counts().plot(kind="bar", title = "Fare Ranges all", ax = f1)

df_surv_fare["fare_ranges"].value_counts().plot(kind="bar", title =  "Survived", ax = f2)

df_nonsurv_fare["fare_ranges"].value_counts().plot(kind="bar", title = "Not Survived", ax = f3)
def name_to_int(df):

    name = df["Name"].values.tolist()

    namelist = []

    for i in name:

        index = 1

        inew = i.split()

        if inew[0].endswith(","):

            index = 1

        elif inew[1].endswith(","):

            index = 2

        elif inew[2].endswith(","):

            index = 3

        namelist.append(inew[index])

    print(set(namelist))

    

    titlelist = []

    

    for i in range(len(namelist)): 

        if namelist[i] == "Lady.":

            titlelist.append("Lady.")

        elif namelist[i] == "Ms.":

            titlelist.append("Ms.")

        elif namelist[i] == "Miss.":

            titlelist.append("Miss.")

        elif namelist[i] == "Dr.":

            titlelist.append("Dr.")

        elif namelist[i] == "Mr.":

            titlelist.append("Mr.")

        elif namelist[i] == "Jonkheer.":

            titlelist.append("Jonkheer.")

        elif namelist[i] == "Col.":

            titlelist.append("Col.")

        elif namelist[i] == "Mrs.":

            titlelist.append("Mrs")

        elif namelist[i] == "Sir.":

            titlelist.append("Sir.")

        elif namelist[i] == "Mlle.":

            titlelist.append("Mlle.")

        elif namelist[i] == "Capt.":

            titlelist.append("Capt.")

        elif namelist[i] == "the":

            titlelist.append("the")

        elif namelist[i] == "Don.":

            titlelist.append("Don.")

        elif namelist[i] == "Master.":

            titlelist.append("Master.")

        elif namelist[i] == "Rev.":

            titlelist.append("Rev.")

        elif namelist[i] == "Mme.":

            titlelist.append("Mme.")

        elif namelist[i] == "Major.":

            titlelist.append("Major.")

        else:

            titlelist.append("sometitle")

    print(set(namelist))

    return titlelist
titlelist = name_to_int(df)

df["titles"] = titlelist

df["titles"].value_counts()

df["titles"].replace(["Ms.","Jonkheer.","the","Don.","Capt.","Sir.","Lady.","Mme.","Col.","Major."],"sometitle", inplace = True)



df["titles"].replace("Mlle.","Miss.", inplace = True)
df["titles"].replace(["Mr.", "Miss.", "Mrs", "Master.", "Dr.", "Rev.", "sometitle"],[0,1,2,3,4,5,6], inplace = True)

df["titles"].astype("int64")





df.drop(["Name"], axis = 1, inplace = True)
df.drop(["SibSp","Parch","Fare","n_fam_mem","actual_fare"], axis = 1, inplace = True)
df.info()

f,ax = plt.subplots(figsize=(6, 5))

sns.heatmap(df.corr(), annot=True,cmap = "Blues", linewidths=.5, fmt= '.2f',ax = ax)

plt.show()