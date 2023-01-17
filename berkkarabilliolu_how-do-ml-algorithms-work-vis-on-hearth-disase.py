# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import missingno as msno # check missing value

import seaborn as sns





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/heart-disease-uci/heart.csv')
data.head()
data.describe()
data.info()
msno.matrix(data)
data.isnull().sum()
# import The libs.

import plotly.graph_objects as go

import plotly.express as px

import plotly.offline as py

from bokeh.io import output_file,show,output_notebook,push_notebook

from bokeh.plotting import figure

from bokeh.models import ColumnDataSource,HoverTool,CategoricalColorMapper

from bokeh.layouts import row,column,gridplot

from bokeh.models.widgets import Tabs,Panel

output_notebook()
data['sex'][data['sex'] == 0] = 'female'

data['sex'][data['sex'] == 1] = 'male'

data.columns = ['age', 'sex', 'chest_pain_degree', 'blood_pressure', 'cholesterol', 'blood_sugar', 'rest_ecg', 'max_heart_rate',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'long_vessels', 'thalassemia', 'target']
plt.figure(figsize = (12,12))

sns.barplot(x = data['target'].value_counts().index,

           y=data['target'].value_counts().values)

plt.xlabel('Target')

plt.ylabel('Frequency')

plt.title('Target Barplot')

plt.show()
data.head()
fig = go.Figure([go.Bar(x=data["chest_pain_degree"], y=data["target"])])

fig.update_layout(title_text="Chest_Pain_degree-Target")

py.iplot(fig, filename="test")
fig = go.Figure([go.Bar(x = data["long_vessels"],y = data["target"])])

fig.update_layout(title_text ="vessels-target")

py.iplot(fig, filename = "test")
fig = go.Figure([go.Bar(x = data['st_depression'],y = data['target'])])

fig.update_layout(title_text = 'Depression-target')

py.iplot(fig,filename = 'test')
fig = go.Figure([go.Bar(x = data['max_heart_rate'],y = data['target'])])

fig.update_layout(title_text = 'max_heart_rate-Target')

py.iplot(fig,filename = 'test')
fig = go.Figure([go.Bar(x=data["sex"], y=data["target"])])

fig.update_layout(title_text="sex-Target")

py.iplot(fig, filename="test") 
data.groupby('target').mean()
fig = go.Figure([go.Bar(x=data["age"], y=data["target"])])

fig.update_layout(title_text="Target by age")

py.iplot(fig, filename="test") 
pd.crosstab(data.age,data.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
#scatter plot (Maximum_heart_rate-Age) Colors(target)

plt.scatter(x=data.age[data.target==1], y=data.max_heart_rate[(data.target==1)], color="black")

plt.scatter(x=data.age[data.target==0], y=data.max_heart_rate[(data.target==0)])

plt.legend(["Disease", "Not Disease"])

plt.xlabel("Age")

plt.ylabel("Maximum Heart Rate")

plt.show()
hist = data.hist(figsize =(10,10))
data.corr()
f,ax = plt.subplots(figsize=(12,12))

sns.heatmap(data.corr(),annot=True,ax=ax)

plt.show()
#Encoding 

data = pd.get_dummies(data,columns = ['sex'],prefix = ['sex'])
#create the test and train

x = data.drop('target',axis=1)

y = data['target']
x['sex_male'] = x['sex_male'].astype(int)

x['sex_female'] = x['sex_female'].astype(int)
#create the model

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=42, stratify=y)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV , StratifiedKFold

from sklearn.model_selection import cross_val_score



# All methods we use

knn_model=KNeighborsClassifier().fit(X_train,y_train)

lr_model=LogisticRegression().fit(X_train,y_train)

rf_model=RandomForestClassifier().fit(X_train,y_train)

lgb_model=LGBMClassifier().fit(X_train,y_train)

xgb_model=XGBClassifier().fit(X_train,y_train)

gbm_model=GradientBoostingClassifier().fit(X_train,y_train)





models=[lr_model,rf_model,lgb_model,gbm_model,xgb_model,knn_model]



sc_fold=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)



for model in models:

    names=model.__class__.__name__

    accuracy=cross_val_score(model,X_train,y_train,cv=sc_fold)

    print("{}s score:{}".format(names,accuracy.mean()))
#import libary's

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV , StratifiedKFold

from sklearn.model_selection import cross_val_score

lgb_model=LGBMClassifier().fit(X_train,y_train)

sc_fold=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

names=lgb_model.__class__.__name__

accuracy=cross_val_score(lgb_model,X_train,y_train,cv=sc_fold)

print("{}s score:{}".format(names,accuracy.mean()))
y = data.target.values

x_data = data.drop(['target'], axis = 1)
# Normalize the dataset

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
#Create the model (train test split)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=0)
#transpose matrices

x_train = x_train.T

y_train = y_train.T

x_test = x_test.T

y_test = y_test.T
#initialize

def initialize(dimension):

    

    weight = np.full((dimension,1),0.01)

    bias = 0.0

    return weight,bias
def sigmoid(z):

    

    y_head = 1/(1+ np.exp(-z))

    return y_head
def forwardBackward(weight,bias,x_train,y_train):

    # Forward

    

    y_head = sigmoid(np.dot(weight.T,x_train) + bias)

    loss = -(y_train*np.log(y_head) + (1-y_train)*np.log(1-y_head))

    cost = np.sum(loss) / x_train.shape[1]

    

    # Backward

    derivative_weight = np.dot(x_train,((y_head-y_train).T))/x_train.shape[1]

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

    gradients = {"Derivative Weight" : derivative_weight, "Derivative Bias" : derivative_bias}

    

    return cost,gradients
#Update parameters

def update(weight,bias,x_train,y_train,learningRate,iteration) :

    costList = []

    index = []

    

    #for each iteration, update weight and bias values

    for i in range(iteration):

        cost,gradients = forwardBackward(weight,bias,x_train,y_train)

        weight = weight - learningRate * gradients["Derivative Weight"]

        bias = bias - learningRate * gradients["Derivative Bias"]

        

        costList.append(cost)

        index.append(i)



    parameters = {"weight": weight,"bias": bias}

    

    print("iteration:",iteration)

    print("cost:",cost)



    plt.plot(index,costList)

    plt.xlabel("Number of Iteration")

    plt.ylabel("Cost")

    plt.show()



    return parameters, gradients
def predict(weight,bias,x_test):

    z = np.dot(weight.T,x_test) + bias

    y_head = sigmoid(z)



    y_prediction = np.zeros((1,x_test.shape[1]))

    

    for i in range(y_head.shape[1]):

        if y_head[0,i] <= 0.5:

            y_prediction[0,i] = 0

        else:

            y_prediction[0,i] = 1

    return y_prediction
def logistic_regression(x_train,y_train,x_test,y_test,learningRate,iteration):

    dimension = x_train.shape[0]

    weight,bias = initialize(dimension)

    

    parameters, gradients = update(weight,bias,x_train,y_train,learningRate,iteration)



    y_prediction = predict(parameters["weight"],parameters["bias"],x_test)

    

    print("Manuel Test Accuracy: {:.2f}%".format((100 - np.mean(np.abs(y_prediction - y_test))*100)))
logistic_regression(x_train,y_train,x_test,y_test,1,100)