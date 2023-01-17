# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Titanic = pd.read_csv('../input/titanic.csv')
class_mf_surviver = pd.pivot_table(Titanic, values='Name', columns=['Survived'], index=['Pclass','Sex'], aggfunc='count')

class_mf_surviver

class_mf_surviver.plot(kind='bar',stacked=True)

plt.show()
mf_surviver = pd.pivot_table(Titanic, values='Name', columns=['Survived'], index=['Sex'], aggfunc='count')

mf_surviver.plot(kind='bar',stacked=True)

plt.show()
class_surviver = pd.pivot_table(Titanic, values='Name', columns=['Survived'], index=['Pclass'], aggfunc='count')

class_surviver.plot(kind='bar',stacked=True)

plt.show()
mf_percent = mf_surviver/Titanic.Sex.count()*100

mf_percent
class_surviver/Titanic.Sex.count()*100


# create age wise bins

bins = [0,5,10,20,40,50,Titanic.Age.max()]

labels = ['Child','Kids','Young','Adult','Mid Age','Old']

Titanic['Age_category'] = pd.cut(Titanic['Age'],bins=bins, labels=labels)
survived_by_Age = pd.pivot_table(Titanic,values = 'Name' ,index='Age_category', columns=['Survived'],aggfunc='count')

survived_by_Age
survived_by_Age.plot(kind='bar')

plt.show()
# analysis on survived passengers



Survived_Pas = Titanic[Titanic.Survived==1]

Survived_by_Age_mf = pd.pivot_table(Survived_Pas, values='Survived', index=['Age_category'], columns='Sex', aggfunc='count')

Survived_by_Age_mf
Survived_by_Age_mf.plot(lw= 2, linestyle = '--', marker='o')

plt.show()
# same analysis in single figure



fig, ax = plt.subplots(1, 4, figsize=(16,5))



ax[0].set_title("class_mf_surviver")

class_mf_surviver.plot(ax = ax[0], kind='bar',stacked=True, alpha=.9)



ax[1].set_title("mf_surviver")

mf_surviver.plot(ax=ax[1], kind='bar', stacked=True,color=['teal','lightseagreen'],alpha=.8)



ax[2].set_title("class_surviver")

class_surviver.plot(ax=ax[2], kind='bar', stacked=True,color=['mediumspringgreen','tomato'],alpha=.9)



ax[3].set_title("survived_by_Age")

survived_by_Age.plot(ax=ax[3], kind='bar', stacked=True,color=['orange','gold'],alpha=.9)



plt.show()