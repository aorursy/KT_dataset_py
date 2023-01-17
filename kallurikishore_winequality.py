# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('../input/winequalityred/winequality-red.csv')
data.head()
data.info()
data.describe()
data.corr()
import seaborn as sm
sm.heatmap(data.corr(),center=0)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor

x=data.drop("quality",axis=1)
y=data.quality
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25,random_state=3)
x_train.shape,y_train.shape
reg= LinearRegression()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)
from sklearn.metrics import accuracy_score

acc=accuracy_score(y_test,y_predict.round(),normalize=True)
print(acc)
model=SGDRegressor()
model.fit(x_train,y_train)
predict=model.predict(x_test)
acc=accuracy_score(y_test,y_predict.round())
acc
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)
Y_predict=model.predict(x_test)
acc=accuracy_score(y_test,y_predict.round())
acc
sm.barplot(x=data['quality'],y=data['fixed acidity'])
sm.countplot(data['quality'])
data[['quality']]=data['quality'].apply(lambda x: 0 if int(x)<6 else 1)
data.quality.value_counts()
y=data['quality']
x=data.drop("quality",axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=20,random_state=2)
from sklearn.linear_model import LogisticRegression
model1=LogisticRegression(max_iter=10000)
model1.fit(x_train,y_train)
model1=model1.score(x_test,y_test)
from sklearn.ensemble import RandomForestClassifier
model2=RandomForestClassifier()
model2.fit(x_train,y_train)
model2=model2.score(x_test,y_test)
from sklearn.tree import DecisionTreeClassifier
model3 = DecisionTreeClassifier()
model3.fit(x_train,y_train)
model3=model3.score(x_test,y_test)
from sklearn.naive_bayes import GaussianNB
model4 = GaussianNB()
model4.fit(x_train,y_train)
model4=model4.score(x_test,y_test)
from  sklearn.neighbors import KNeighborsClassifier
model5=KNeighborsClassifier()
model5.fit(x_train,y_train)
model5=model5.score(x_test,y_test)
final_results=pd.DataFrame({'models':['LogisticRegression','RandomForset','Dessiontree','GaussianNB','KNeighborsClassifier'],'accuracy_score':[model1,model2,model3,model4,model5]})
sm.barplot(x=final_results['models'],y=final_results['accuracy_score'])