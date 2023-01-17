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

import sys

#!conda install --yes --prefix {sys.prefix} plotly

import os

import matplotlib.pyplot as plt#visualization

from PIL import  Image

%matplotlib inline

import pandas as pd

import seaborn as sns#visualization

import itertools

import warnings

warnings.filterwarnings("ignore")

import io

import plotly.offline as py#visualization

py.init_notebook_mode(connected=True)#visualization

import plotly.graph_objs as go #visualization

import plotly.tools as tls#visualization

import plotly.figure_factory as ff#visualization

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
dataset.head()
print("Missing Values:",dataset.isnull().sum())
print("Unique Values:",dataset.nunique().sum())
### Data Manipulation



#Replacing spaces with null values in total charges column



dataset['TotalCharges'] = dataset['TotalCharges'].replace(" ",np.nan)

print("Missing Values:",dataset.isnull().sum())
#replace 'No internet service' to No for the following columns

replace_cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',

                'TechSupport','StreamingTV', 'StreamingMovies']

for i in replace_cols : 

    dataset[i]  = dataset[i].replace({'No internet service' : 'No'})
dataset.head()
#Separating churn and non churn customers

churn     = dataset[dataset["Churn"] == "Yes"]

not_churn = dataset[dataset["Churn"] == "No"]
# DATA Analysis

#labels

lab = dataset["Churn"].value_counts().keys().tolist()

#values

val = dataset["Churn"].value_counts().values.tolist()



trace = go.Pie(labels = lab ,

               values = val ,

               marker = dict(colors =  [ 'royalblue' ,'lime'],

                             line = dict(color = "white",

                                         width =  1.3)

                            ),

               rotation = 90,

               hoverinfo = "label+value+text",

               hole = .5

              )

layout = go.Layout(dict(title = "Customer attrition in data",

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                       )

                  )



data = [trace]

fig = go.Figure(data = data,layout = layout)

py.iplot(fig)
dataset['gender'].replace(['Male','Female'],[0,1],inplace=True)

dataset['Partner'].replace(['Yes','No'],[1,0],inplace=True)

dataset['Dependents'].replace(['Yes','No'],[1,0],inplace=True)

dataset['PhoneService'].replace(['Yes','No'],[1,0],inplace=True)

dataset['MultipleLines'].replace(['No phone service','No', 'Yes'],[0,0,1],inplace=True)

dataset['InternetService'].replace(['No','DSL','Fiber optic'],[0,1,2],inplace=True)

dataset['OnlineSecurity'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)

dataset['OnlineBackup'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)

dataset['DeviceProtection'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)

dataset['TechSupport'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)

dataset['StreamingTV'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)

dataset['StreamingMovies'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)

dataset['Contract'].replace(['Month-to-month', 'One year', 'Two year'],[0,1,2],inplace=True)

dataset['PaperlessBilling'].replace(['Yes','No'],[1,0],inplace=True)

dataset['PaymentMethod'].replace(['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'],[0,1,2,3],inplace=True)

dataset['Churn'].replace(['Yes','No'],[1,0],inplace=True)

dataset



# Dropping Customer Id since not giving any inference

dataset.pop('customerID')



dataset.info() 
### Pearson Co-relation



#correlation

correlation = dataset.corr()

#tick labels

matrix_cols = correlation.columns.tolist()

#convert to array

corr_array  = np.array(correlation)



#Plotting

trace = go.Heatmap(z = corr_array,

                   x = matrix_cols,

                   y = matrix_cols,

                   colorscale = "Viridis",

                   colorbar   = dict(title = "Pearson Correlation coefficient",

                                     titleside = "right"

                                    ) ,

                  )



layout = go.Layout(dict(title = "Correlation Matrix for variables",

                        autosize = False,

                        height  = 720,

                        width   = 800,

                        margin  = dict(r = 0 ,l = 210,

                                       t = 25,b = 210,

                                      ),

                             yaxis   = dict(tickfont = dict(size = 9)),

                        xaxis   = dict(tickfont = dict(size = 9))

                       )

                  )



data = [trace]

fig = go.Figure(data=data,layout=layout)

py.iplot(fig)
dataset.pop('TotalCharges')

dataset.dropna(subset=['MonthlyCharges'], inplace=True)

dataset.dropna(subset=['tenure'], inplace=True)

dataset.head()
#### MODEl BUILDING



from sklearn.model_selection import train_test_split 

train, test = train_test_split(dataset, test_size = 0.25)

 

train_y = train['Churn']

test_y = test['Churn']

 

train_x = train

train_x.pop('Churn')

test_x = test

test_x.pop('Churn')
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report

 

logisticRegr = LogisticRegression()

logisticRegr.fit(train_x,train_y)
test_y_pred = logisticRegr.predict(test_x)

confusion_matrix = confusion_matrix(test_y, test_y_pred)

print('Intercept: ' + str(logisticRegr.intercept_))

print('Regression: ' + str(logisticRegr.coef_))

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logisticRegr.score(test_x, test_y)))
#### Through decision tree



from sklearn import tree

from sklearn import tree

 

# Create each decision tree (pruned and unpruned)

decisionTree_unpruned = tree.DecisionTreeClassifier()

decisionTree = tree.DecisionTreeClassifier(max_depth = 4)

 

# Fit each tree to our training data

decisionTree_unpruned = decisionTree_unpruned.fit(X=train_x, y=train_y)

decisionTree = decisionTree.fit(X=train_x, y=train_y)
test_y_pred_dt = decisionTree.predict(test_x)

print('Accuracy of decision tree classifier on test set: {:.2f}'.format(decisionTree.score(test_x, test_y)))
###through random forest 

from sklearn.ensemble import RandomForestClassifier

randomForest = RandomForestClassifier()

randomForest.fit(train_x, train_y)

print('Accuracy of random forest classifier on test set:{:.2f} '.format(randomForest.score(test_x, test_y)))
