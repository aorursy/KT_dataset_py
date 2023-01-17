# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import tensorflow as tf



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
input_ds = pd.read_csv("/kaggle/input/xAPI-Edu-Data/xAPI-Edu-Data.csv")

input_ds.gender = input_ds.gender.replace({'F': 'Female', 'M': 'Male'})
print("Total samples %d" % len(input_ds))

print("Columns:\n\t%s" % "\n\t".join(input_ds.columns))
input_ds.gender.value_counts().plot(kind="bar")
all_stages = input_ds.StageID.value_counts()



male_stages = input_ds[input_ds.gender == "Male"].StageID.value_counts()

female_stages = input_ds[input_ds.gender == "Female"].StageID.value_counts()



p_male = plt.bar(all_stages.index, male_stages)

p_female = plt.bar(all_stages.index, female_stages, bottom=male_stages)



plt.legend((p_male[0], p_female[0]), ('Male', 'Female'))
input_ds.GradeID.value_counts().plot(kind="bar")
print("High school")

print(input_ds[input_ds.StageID == "HighSchool"].GradeID.value_counts())

print("Middle school")

print(input_ds[input_ds.StageID == "MiddleSchool"].GradeID.value_counts())

print("Lower level")

print(input_ds[input_ds.StageID == "lowerlevel"].GradeID.value_counts())

# Why there's G-07 in lower level?
def plot_bar(ds, subplot, gender):

    subplot.set_title(gender)

    selected_topics = ds[input_ds.gender == gender].Topic.value_counts()

    subplot.barh(selected_topics.index, selected_topics)





def plot_topics(ds):

    figure = plt.figure(figsize=(15, 10))



    gs = figure.add_gridspec(2, 2)

    subplot = figure.add_subplot(gs[0, :])

    subplot.set_title("Total")



    index = ds.Topic.value_counts().index

    values = ds.Topic.value_counts()

    subplot.barh(index, values)



    subplot = figure.add_subplot(gs[1, 0])

    plot_bar(ds, subplot, 'Male')



    subplot = figure.add_subplot(gs[1, 1])

    plot_bar(ds, subplot, 'Female')



plot_topics(input_ds)
plot_topics(input_ds[input_ds.StageID == "lowerlevel"])
plot_topics(input_ds[input_ds.StageID == "MiddleSchool"])
plot_topics(input_ds[input_ds.StageID == "HighSchool"])
male_relation = input_ds[input_ds.gender == "Male"].Relation.value_counts()

female_relation = input_ds[input_ds.gender == "Female"].Relation.value_counts()



index = female_relation.index

p_male = plt.bar(index, male_relation)

p_female = plt.bar(index, female_relation, bottom=male_relation)

plt.legend((p_male[0], p_female[0]), ('Male', 'Female'))
above_7 = input_ds[input_ds.StudentAbsenceDays == "Above-7"]

under_7 = input_ds[input_ds.StudentAbsenceDays == "Under-7"]

classes = ["L", "M", "H"]

def plot_classes(first_series, second_series, titles):

    first = [first_series["L"], first_series["M"], first_series["H"]]

    first_bar = plt.bar(classes, first)

    second_bar = plt.bar(classes, [second_series["L"], second_series["M"], second_series["H"]], bottom=first)

    plt.legend((first_bar[0], second_bar[0]), titles)
plt.figure(figsize=(10, 10))

plt.subplot(311)

plt.title("Marks by gender")

males = input_ds[input_ds.gender == "Male"].Class.value_counts()

females = input_ds[input_ds.gender == "Female"].Class.value_counts()

plot_classes(males, females, ('Male', 'Female'))



plt.subplot(312)

plt.title("Marks by semester")

males = input_ds[input_ds.Semester == "F"].Class.value_counts()

females = input_ds[input_ds.Semester == "S"].Class.value_counts()

plot_classes(males, females, ('First', 'Second'))



plt.subplot(313)

plt.title("Marks by absence")

above_7 = input_ds[input_ds.StudentAbsenceDays == "Above-7"]

under_7 = input_ds[input_ds.StudentAbsenceDays == "Under-7"]

plot_classes(above_7.Class.value_counts(), under_7.Class.value_counts(), ('Above 7 absence days', 'Below 7 absence days'))
sns.distplot(input_ds[input_ds["Class"] == "L"].raisedhands, kde=False, label="L")

sns.distplot(input_ds[input_ds["Class"] == "M"].raisedhands, kde=False, label="M")

sns.distplot(input_ds[input_ds["Class"] == "H"].raisedhands, kde=False, label="H")

plt.legend()

plt.title("Raised hands distribution")
sns.distplot(input_ds[input_ds["Class"] == "L"].AnnouncementsView, kde=False, label="L")

sns.distplot(input_ds[input_ds["Class"] == "M"].AnnouncementsView, kde=False, label="M")

sns.distplot(input_ds[input_ds["Class"] == "H"].AnnouncementsView, kde=False, label="H")

plt.legend()

plt.title("Viewing announcements distribution")
sns.distplot(input_ds[input_ds["Class"] == "L"].Discussion, kde=False, label="L")

sns.distplot(input_ds[input_ds["Class"] == "M"].Discussion, kde=False, label="M")

sns.distplot(input_ds[input_ds["Class"] == "H"].Discussion, kde=False, label="H")

plt.legend()

plt.title("Discussion participation")
sns.distplot(input_ds[input_ds["Class"] == "L"].VisITedResources, kde=False, label="L")

sns.distplot(input_ds[input_ds["Class"] == "M"].VisITedResources, kde=False, label="M")

sns.distplot(input_ds[input_ds["Class"] == "H"].VisITedResources, kde=False, label="H")

plt.legend()

plt.title("Visited resoursed")
sns.scatterplot(input_ds.AnnouncementsView, input_ds.raisedhands, hue=input_ds["Class"])

plt.title("Raised hands to announcements view")
sns.lmplot('AnnouncementsView', 'raisedhands', hue="Class", data=input_ds)

plt.title("Raised hands to announcements view")
sns.jointplot('raisedhands', 'AnnouncementsView', data=input_ds, kind="kde")