# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

import random

from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/glass.csv')

df.head(10)
df.info()
df.describe()
#melting dataframe for value plot

df_melt = df.melt("Type", var_name="element")

#creating a log-based scale to see things better

df_melt.value = np.log10(df_melt.value)

fig, ax = plt.subplots()

fig.set_size_inches(20, 20)

sns.swarmplot(x="element", y="value", hue="Type", data=df_melt)
fig, ax = plt.subplots(1,5)

fig.set_size_inches(50, 20)

elements = list(df.columns)

for i in range(5):

    t = sns.boxplot(x="Type", y=elements[i], data=df, ax=ax[i])

    t.set_xlabel("Type",fontsize=30)

    t.axes.set_title(elements[i],fontsize=50)

    t.tick_params(labelsize=20)
fig, ax = plt.subplots(1,4)

fig.set_size_inches(50, 20)

elements = list(df.columns)

for i in range(5,9):

    t = sns.boxplot(x="Type", y=elements[i], data=df, ax=ax[i-5])

    t.set_xlabel("Type",fontsize=30)

    t.axes.set_title(elements[i],fontsize=50)

    t.tick_params(labelsize=20)
x, y = df.iloc[:,:9], df.iloc[:,9:]

print(list(x.columns), list(y.columns))
#there is a really annoying warning message that gets printed so I've just hidden the output

results=[]

random.seed(0)

randints = [random.randint(0,1000) for i in range(100)]

for i in range(100):

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.1, random_state = randints[i])

    clf = RandomForestClassifier(random_state = randints[i])

    clf.fit(x_train, y_train)

    predict=clf.predict(x_test)

    results.append(accuracy_score(y_test,predict))    
print('mean accuracy  ', round(np.mean(results)*100,2),'%')

print('standard deviation  ', round(np.std(results)*100,2),'%')
plt.hist(results, bins=[0.05 * i + 0.2 for i in range(20)])