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
data=pd.read_csv("/kaggle/input/uci-turkiye-student-evaluation-data-set/turkiye-student-evaluation_generic.csv")

data.head()
%matplotlib inline

import seaborn as sns

sns.countplot(data=data,x="class")
data.isnull().isnull().sum()
y=data["class"]

x=data.iloc[:,2:34]
from tpot import TPOTClassifier

from sklearn.model_selection import train_test_split

tpot = TPOTClassifier(verbosity=2,max_time_mins=2)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1923)

tpot.fit(x_train, y_train)

print(tpot.score(x_test, y_test))
from sklearn.dummy import DummyClassifier

baseline=DummyClassifier(strategy="most_frequent")

baseline.fit(x_train,y_train)





import sklearn.metrics as metrik

ypred=baseline.predict(x_test)

metrik.accuracy_score(y_true=y_test,y_pred=ypred)
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

ypred=rfc.predict(x_test)

metrik.accuracy_score(y_true=y_test,y_pred=ypred)
rfc=RandomForestClassifier(max_depth=15)

rfc.fit(x_train,y_train)

ypred=rfc.predict(x_test)

metrik.accuracy_score(y_true=y_test,y_pred=ypred)
rfc=RandomForestClassifier(max_depth=2)

rfc.fit(x_train,y_train)

ypred=rfc.predict(x_test)

metrik.accuracy_score(y_true=y_test,y_pred=ypred)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda=LinearDiscriminantAnalysis()

x_train_lda=lda.fit_transform(x_train,y_train)

x_train_lda.shape
x_test_lda=lda.transform(x_test)
from tpot import TPOTClassifier

from sklearn.model_selection import train_test_split

tpot = TPOTClassifier(verbosity=2,max_time_mins=3)

tpot.fit(x_train_lda, y_train)

print(tpot.score(x_test_lda, y_test))
rfc=RandomForestClassifier(max_depth=15)

rfc.fit(x_train_lda,y_train)

ypred=rfc.predict(x_test_lda)

metrik.accuracy_score(y_true=y_test,y_pred=ypred)
lda=LinearDiscriminantAnalysis(n_components=2)

xplot=lda.fit_transform(x,y)

xplot.shape
forplot=pd.DataFrame(xplot)

forplot.columns=["bir","iki"]
forplot["class"]=y
import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))

sns.scatterplot(data=forplot,x="bir",y="iki",hue="class")
tpot = TPOTClassifier(verbosity=2,max_time_mins=135)

tpot.fit(x_train_lda, y_train)

print(tpot.score(x_test_lda, y_test))