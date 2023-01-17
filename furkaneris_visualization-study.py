# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
student = pd.read_csv("../input/StudentsPerformance.csv")
student.head()
student.info()
#According to students points,seperate letter grade 

def letter_point(point):

    if point < 30:

        return "FF"

    elif point >= 30 and point < 40:

        return "DD"

    elif point >= 40 and point < 50:

        return "DC"

    elif point >= 50 and point < 60:

        return "CC"

    elif point >= 60 and point < 70:

        return "CB"

    elif point >= 70 and point < 80:

        return "BB"

    elif point >= 80 and point < 90:

        return "BA"

    elif point >= 90:

        return "AA"

student["mathletter"] = student["math score"].apply(letter_point)

student["readingletter"] = student["reading score"].apply(letter_point)

student["writingletter"] = student["writing score"].apply(letter_point)
# According to letter grade,seperate 3 class; successful,unsuccessful,conditional

def category(letter):

    if letter == "FF":

        return "unsuccessful"

    elif letter == "DC" or letter == "DD":

        return "conditional"

    else :

        return "successful"

student["math_result"] = student["mathletter"].apply(category)

student["reading_result"] = student["readingletter"].apply(category)

student["writing_result"] = student["writingletter"].apply(category)

student["test preparation course"].value_counts()
sns.countplot(student["test preparation course"])

plt.show()
test_pre = student.groupby("test preparation course")

test_pre_mean = test_pre.mean()

test_pre_mean
plt.figure(figsize = (10,10))

plt.subplot(3,1,1)

sns.swarmplot(x = "test preparation course",y = "math score" ,hue = "math_result",data = student)

plt.subplot(3,1,2)

sns.swarmplot(x = "test preparation course",y = "reading score" ,hue = "reading_result",data = student)

plt.subplot(3,1,3)

sns.swarmplot(x = "test preparation course",y = "writing score" ,hue = "writing_result",data = student)

plt.show()
student["race/ethnicity"].value_counts()
plt.figure(figsize = (15,5))

plt.subplot(1,3,1)

sns.countplot(student["mathletter"])

plt.subplot(1,3,2)

sns.countplot(student["readingletter"])

plt.subplot(1,3,3)

sns.countplot(student["writingletter"])

plt.show()
plt.figure(figsize = (24,20))

plt.subplot(1,3,1)

plt.pie(x = student.mathletter.value_counts().values,autopct = "%.1f%%",labels = student.mathletter.value_counts().index)

plt.title("MATH LETTER")

plt.subplot(1,3,2)

plt.pie(x = student.readingletter.value_counts().values,autopct = "%.1f%%",labels = student.readingletter.value_counts().index)

plt.title("READİNG LETTER")

plt.subplot(1,3,3)

plt.pie(x = student.writingletter.value_counts().values,autopct = "%.1f%%",labels = student.writingletter.value_counts().index)

plt.title("WRİTİNG LETTER")

plt.show()
sns.countplot(student["race/ethnicity"])

plt.show()
plt.figure(figsize = (10,20))

plt.subplot(3,1,1)

sns.boxplot(x = "race/ethnicity",y = "math score",data = student)

plt.subplot(3,1,2)

sns.boxplot(x = "race/ethnicity",y = "reading score",data = student)

plt.subplot(3,1,3)

sns.boxplot(x = "race/ethnicity",y = "writing score",data = student)

plt.show()
student["average_point"] = (student["math score"] + student["reading score"] + student["writing score"])/3

student["average_letter"] = student["average_point"].apply(letter_point)

student["general_result"] = student["average_letter"].apply(category)
student.head()
student.groupby("gender").mean()
plt.figure(figsize = (15,7))

plt.subplot(1,3,1)

sns.barplot(x = student.gender,y = student["math score"])

plt.title("MATH")

plt.subplot(1,3,2)

sns.barplot(x = student.gender,y = student["reading score"])

plt.title("READİNG")

plt.subplot(1,3,3)

sns.barplot(x = student.gender,y = student["writing score"])

plt.title("WRİTİNG")

plt.show()
#How about genaral result

sns.barplot(x = student.gender,y = student.average_point)

plt.show()
x = student["math score"]

y = student["reading score"]

sns.jointplot(x = x,y = y,kind = "kde")
x = student["math score"]

y = student["writing score"]

sns.jointplot(x = x,y = y,kind = "kde")
x = student["reading score"]

y = student["writing score"]

sns.jointplot(x = x,y = y,kind = "kde")
student.corr()
sns.heatmap(student.corr(),annot = True)

plt.show()