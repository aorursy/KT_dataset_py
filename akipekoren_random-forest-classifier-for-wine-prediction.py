# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/winequality-red.csv") # reading the data

data.head(10)
data.describe() # mean and std are really important two parameter for machine learning
data.info()
data["quality"].unique()
data["quality"].value_counts()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
quality_list = []

for each in data.quality:

    if each > 5:

        quality_list.append("Good")

    elif each <= 5:

        quality_list.append("Bad")

quality_list

    
data["evaluation"] = quality_list

ax = sns.countplot(data["evaluation"],label="Count")

Good, Bad = data["evaluation"].value_counts()

print("Number of good wine: ", Good)

print("Number of bad wine: ", Bad)
y = data["evaluation"]

x = data.drop(["evaluation"], axis = 1)



data_normalized = (x - x.min())/(x.max()-x.min()) # normalization

data_normalized



data = pd.concat([y, data_normalized], axis = 1)



data = pd.melt(data,id_vars="evaluation",

                    var_name="features",

                    value_name='value')

sns.violinplot(x="features", y="value", hue="evaluation",

               data=data,split=True, inner="quart")

plt.xticks(rotation=90)

plt.figure(figsize=(10,10))

sns.boxplot(x="features", y="value", hue="evaluation", data=data)

plt.xticks(rotation=90)


sns.set(style="darkgrid", color_codes=True)

a = sns.jointplot(x.loc[:,"total sulfur dioxide"], x.loc[:,"free sulfur dioxide"], 

                  data = x, kind="reg", height=8,color="#ce1414")

a.annotate(stats.pearsonr)

plt.show()
sns.set(style="white")

df = x.loc[:,['alcohol','pH','density']]

g = sns.PairGrid(df, diag_sharey=False)

g.map_lower(sns.kdeplot, cmap="Blues_d")

g.map_upper(plt.scatter)

g.map_diag(sns.kdeplot, lw=3)
import time

sns.set(style="whitegrid", palette="muted")





plt.figure(figsize=(10,10))

tic = time.time()

sns.swarmplot(x="features", y="value", hue="evaluation", data=data)

plt.xticks(rotation=90)
x = x.drop(["quality"],axis=1) # lets remove quality in the list and make ready for prediction.

x                              # because if we put quality into data, prediction becomes approximately

                                #100% due to quality values(Easy to predict).
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score,confusion_matrix





x_train, x_test, y_train, y_test = train_test_split(

    x,y,test_size = 0.3, random_state = 42)



clf_rf = RandomForestClassifier()      

clr_rf = clf_rf.fit(x_train,y_train)



ac = clr_rf.score(x_test, y_test)

print('Accuracy is: ',ac)

cm = confusion_matrix(y_test,clf_rf.predict(x_test))

sns.heatmap(cm,annot=True,fmt="d")