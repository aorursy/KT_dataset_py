# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/world-happiness-report-2020/WHR20_DataForFigure2.1.csv")

data.head()

data.info()
data=data.drop(['upperwhisker','lowerwhisker'],axis=1)

data.columns
import seaborn as sns

import matplotlib.pyplot as plt
cormat=data.corr()

top_cor=cormat.index

plt.figure(figsize=(40,40))

g=sns.heatmap(data[top_cor].corr(),annot=True,cmap='RdYlGn')
#data=data.drop(['Standard error of ladder score',

                #'Generosity','Perceptions of corruption', 

                #'Ladder score in Dystopia',

                #'Explained by: Generosity'],axis=1)

data.info()
x=data.iloc[:,3:13]

y=data.iloc[:,2:3]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=0)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_train,y_train)
print("score of training is {:.2f}".format(lr.score(X_train,y_train)))

print("score of test is {:.2f}".format(lr.score(X_test,y_test)))