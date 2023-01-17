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
# importing all neccessary libraries for Exploratory Data Analysis and visualization
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objects as go
import plotly.express as px
iris = pd.read_csv("../input/iris/Iris.csv")

#getting idea of dataset
iris.head(2),iris.shape,iris.dtypes
iris.drop('Id',axis=1,inplace=True)
iris.columns  # to check that the column 'Id' is Dropped
labels=iris['Species'].unique()
values=iris['Species'].value_counts()
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
py.iplot(fig)
sns.pairplot(iris,hue='Species')

from plotly.subplots import make_subplots
fig = make_subplots(
    rows=2, cols=2)
fig.add_trace(go.Violin(x=iris['Species'],y=iris['PetalWidthCm'],box_visible=True,meanline_visible=True,name='PetalWidth'),row=1,col=1)
fig.add_trace(go.Violin(x=iris['Species'],y=iris['PetalLengthCm'],box_visible=True,meanline_visible=True,name='PetalLength'),row=1,col=2)
fig.add_trace(go.Violin(x=iris['Species'],y=iris['SepalWidthCm'],box_visible=True,meanline_visible=True,name='SepalWidth'),row=2,col=1)
fig.add_trace(go.Violin(x=iris['Species'],y=iris['SepalLengthCm'],box_visible=True,meanline_visible=True,name='SepalLength'),row=2,col=2)
fig.update_layout(height=600, width=1000, title_text="Length and Width variation of Sepals and Petals according to the species")
py.iplot(fig)
#heatmap
sns.heatmap(iris.corr(),annot=True)
#importing neccessary libraries and fuctions for machine learning
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import warnings  
warnings.filterwarnings('ignore')
#splitting the data into training data and testing data
x=iris.drop('Species',axis=1)
y=iris['Species']
x_train, x_test, y_train, y_test = train_test_split(x, y)
lr_model = LogisticRegression() #select the algorithm
lr_model.fit(x_train,y_train) # we train the algorithm with the training data and the training output
lr_predict = lr_model.predict(x_test) #now we pass the testing data to the trained algorithm
print('Accuracy obtained using Logistic Regression - ',round(accuracy_score(lr_predict,y_test)*100,2),'%') #now we check the accuracy of the algorithm. 
#we pass the predicted output by the model and the actual output
knn_model = KNeighborsClassifier()
knn_model.fit(x_train,y_train)
knn_predict=knn_model.predict(x_test)
print('Accuracy obtained using K Nearest Neighbours (KNN) - ',round(accuracy_score(knn_predict,y_test)*100,2),'%')
dt_model=DecisionTreeClassifier()
dt_model.fit(x_train,y_train)
dt_predict=dt_model.predict(x_test)
print('Accuracy obtained using Decision Tree - ',round(accuracy_score(dt_predict,y_test)*100,2),'%')
xgb_model = XGBClassifier()
xgb_model.fit(x_train, y_train)
predict=xgb_model.predict(x_test)
accuracy_score(predict,y_test)
print('Accuracy obtained using XGBoost - ',round(accuracy_score(predict,y_test)*100,2),'%')
svc_model = svm.SVC()
svc_model.fit(x_train,y_train)
svc_predict=svc_model.predict(x_test)
print('Accuracy obtained using Support Vector Classifier (SVC) -',round(accuracy_score(svc_predict,y_test)*100,2),'%')
x1=iris.drop(['Species','PetalWidthCm'],axis=1)  # dropping PetalWidthCm also
y1=iris['Species'] 
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1)
xgb_model = XGBClassifier()
xgb_model.fit(x1_train, y1_train)
predict=xgb_model.predict(x1_test)
accuracy_score(predict,y1_test)
print('Accuracy obtained using XGBoost - ',round(accuracy_score(predict,y1_test)*100,2),'%')
