# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
pubg = pd.read_csv('../input/train.csv')
pubg.head()
import scipy 
from scipy import stats
import seaborn as sb
import matplotlib.pyplot as plt
%matplotlib inline
sb.heatmap(pubg.isnull())

sb.boxplot(pubg.killPlace)
cor = pubg.corr()
round(cor,3)
plt.figure(figsize=(30,30))
sb.heatmap(cor,cmap="Blues",annot=True)
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
pubg.columns
X = pubg.iloc[:,:25]
y = pubg.iloc[:,25]
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size = 0.3,random_state = 10)
len(xtrain)
model1 = LinearRegression()
model1.fit(xtrain,ytrain)
ypred = model1.predict(xtest)
r2_score(ytest,ypred)
pubg_test = pd.read_csv("../input/test.csv")
final_output = model1.predict(pubg_test)
final_output = pd.DataFrame(final_output)
final_output.head()
pubg_test.head()
Id_df = pubg_test.loc[:,'Id']
Id_df.head()
final_output1 = pd.concat([Id_df,final_output],axis=1)
final_output1.head()
final_output1.columns = ['Id','winPlacePerc']
final_output1.head()
final_output1.to_csv('final_output2.csv',index=False)
