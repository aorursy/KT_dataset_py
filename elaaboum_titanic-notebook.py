# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
train.head(5)
train.info()
{'female' : train[train["Sex"]=="female"]["Survived"].mean(), 'male' : train[train["Sex"]=="male"]["Survived"].mean()}
{'1' : train[train["Pclass"]==1]["Survived"].mean(), '2' : train[train["Pclass"]==2]["Survived"].mean() , '3' : train[train["Pclass"]==3]["Survived"].mean()}
train.groupby([ "Pclass", "Sex"])["Survived"].count()
fille_3 = train[(train["Sex"]=="female") & (train["Pclass"]==3)]
fille_3[fille_3["Age"].isnull()]["Survived"].count()
train["Name"].head(10)
a = "mouad, elaaboudi, Mr"
name = train['Name']
train["family"] = name.apply(lambda x: x[:x.index(",")])
train.info()
families = train.groupby("family").mean()
families
