import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.subplots import make_subplots

import plotly.express as px

import plotly.graph_objects as go
%config InlineBackend.figure_format = 'retina'
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.head()        
# object to float



df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors= 'coerce')
df.isnull().sum()
df.dropna(inplace = True)
# Gender

values = df['gender'].value_counts()

names = df['gender'].value_counts().index



fig = px.pie(values= values, names= names)

fig.update_traces(textinfo = 'percent + label', 

                  textfont_size= 12

                  ,marker=dict(line=dict(color='#000000', width = 1.2)))

fig.update_layout(title = '<b>Gender')

fig.show()
# Gender vs Senior Citizen

## Male

v1 = df.query('gender == "Male"')['SeniorCitizen'].value_counts()



## Female

v2 = df.query('gender == "Female"')['SeniorCitizen'].value_counts()



fig = make_subplots(1, 2, specs=[[{'type' : 'domain'}, 

                                   {'type' : 'domain'}]], 

                   subplot_titles= ['Male', 'Female'])



fig.add_trace(go.Pie(values = v1, labels = ['Adult', 'Senior'], 

                     pull = [0, 0.2], scalegroup = 'one'), 

              1, 1)



fig.add_trace(go.Pie(values = v2, labels = ['Adult', 'Senior'], 

                     pull = [0, 0.2], scalegroup = 'one'), 

              1, 2)



fig.update_traces(textinfo='label+percent', textfont_size=12,

                  marker=dict(line=dict(color='#000000', width=1.2)))



fig.update_layout(title = '<b>Senior ratio per gender')

fig.show()
#Churn ratio per Senior citizen

fig = px.histogram(df, x = 'gender', color= 'Churn', barmode='group', facet_col= 'SeniorCitizen')

fig.update_layout(title = '<b>Churn ratio per Senior citizen')
# Partner & Dependents

fig = px.histogram(df, x ='Partner', color= 'Dependents', facet_col= 'Dependents', histnorm= 'probability density')

fig.update_layout(title = '<b>Relation between Partner and Dependents')

fig.show()
# PaperlessBilling

fig = px.histogram(data_frame=df, color='PaperlessBilling', x = 'PaymentMethod',

                   histfunc= 'avg', histnorm= 'probability density', 

                   facet_row= 'PaperlessBilling', height= 600)

fig.update_layout(title = '<b>Paperless billing')

fig.show()
# Contract period

fig= px.histogram(df, x = 'Contract', color= 'Churn', histfunc= 'avg', facet_col = 'Churn')

fig.update_layout(title = '<b>Contrace period')

fig.show()
# Monthly Charges

fig = px.scatter(data_frame= df, y = 'MonthlyCharges', x = 'tenure', color='MonthlyCharges', 

           facet_col= 'Contract', facet_row= 'Churn',

                color_continuous_scale= 'rdylbu')

fig.show()
# Tenure

fig = px.histogram(df, x = 'tenure', color = 'Churn', barmode= 'group', title = '<b>Tenure')

fig.show()
# Total Chargers

fig = px.scatter(data_frame= df, y = 'TotalCharges', x = 'tenure', color='TotalCharges', facet_col= 'Churn',

                color_continuous_scale= 'rdylbu')

fig.show()
# Services

services = ['PhoneService', 'MultipleLines', 'InternetService',

       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',

       'StreamingTV', 'StreamingMovies']



df_services = df[services]
for service in services:

   df_services[service] = df_services[service].map(lambda x :0 if x == 'No' else 1)



df_services.head()
df['Services_sum'] = df_services.sum(axis=1)

df.head()
# sum of services by customer

values = df_services.sum(axis = 1).value_counts()

names = df_services.sum(axis = 1).value_counts().index



fig = px.pie(values= values, names= names, title= 'Sum of survices using')

fig.update_traces(textinfo = 'percent + label',

                 textfont_size = 12,

                  textposition = 'inside',

                 marker = dict(line = dict(color = '#000000', width = 1.2)))

fig.show()

print('Average of Services usage : %d' %df['Services_sum'].mean())
# Which service is most used?

values = df_services.sum().sort_values(ascending= False)

names = df_services.sum().sort_values(ascending= False).index



fig = px.pie(values= values, names= names, 

             title = '<b>Which service is most used?')

fig.update_traces(textinfo = 'percent + label',

                 textfont_size = 12,

                  textposition = 'inside',

                 marker = dict(line = dict(color = '#000000', width = 1.2)),

                 pull = [0.2, 0.1, 0, 0, 0, 0, 0, 0,0])

fig.show()
order = df['Services_sum'].value_counts().index.sort_values().tolist()



fig = px.violin(data_frame= df, x = 'Services_sum', y = 'TotalCharges', color = 'Services_sum', box = True,

               category_orders={'Services_sum' : order})

fig.update_layout(title = '<b>Total Charges per Services usage')

fig.show()

order = df['Services_sum'].value_counts().index.sort_values().tolist()



fig = px.violin(data_frame= df, x = 'Services_sum', y = 'MonthlyCharges', color = 'Services_sum', box = True,

               category_orders={'Services_sum' : order})

fig.update_layout(title = '<b>Monthly Charges per Services usage')

fig.show()
# Vaild Features Selection



features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',

       'tenure', 'Contract', 'MonthlyCharges', 'TotalCharges', 

            'Churn', 'Services_sum']



df_f = df[features]

df_f.head()
df_f['Churn'].replace('Yes', 1, inplace = True)

df_f['Churn'].replace('No', 0, inplace = True)
## One-hot encoding



df_encoding = pd.get_dummies(df_f[features])

df_encoding.head()
px.bar(df_encoding.corr()['Churn'].sort_values(ascending = False))
# Split train/test dataset



X = df_encoding.drop(columns= 'Churn')

y = df_encoding['Churn']



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 15)



print(X_train.shape, X_test.shape, y_train.shape, y_test.shape )
# Logistic Regression

from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(random_state=15, n_jobs = -1).fit(X_train, y_train)

clf.score(X_test, y_test)
# Decision Tree

from sklearn.tree import DecisionTreeClassifier



clf_score = []



for max_depth in range(2, 10):

    clf = DecisionTreeClassifier(max_depth = max_depth, random_state = 15).fit(X_train, y_train)

    clf_score.append(clf.score(X_test, y_test))



# fig = px.line(x = range(2, 10), y = tree_score)

g = plt.plot(range(2, 10), clf_score)

g = plt.plot(clf_score.index(max(clf_score)) + min(range(2,10)), max(clf_score), 'o')

g = plt.title('DecisionTreeClassifier')

print(max(clf_score))
# Nearest Neighbors Classification



from sklearn.neighbors import KNeighborsClassifier



clf_score = []



for n_neighbors in range(2, 10):

    clf = KNeighborsClassifier(n_neighbors = n_neighbors, 

                           n_jobs= -1).fit(X_train, y_train)

    clf_score.append(clf.score(X_test, y_test))



    

g = plt.plot(range(2, 10), clf_score)

g = plt.plot(clf_score.index(max(clf_score)) + min(range(2,10)), max(clf_score), 'o')

g = plt.title('Nearest Neighbors Classification')

print(max(clf_score))
# Support Vector Machines



from sklearn.svm import SVC



clf = SVC().fit(X_train, y_train)



clf.score(X_test, y_test)
# RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

clf_score = []



for max_depth in range(2, 15):

    clf = RandomForestClassifier(max_depth = max_depth, random_state = 15).fit(X_train, y_train)

    clf_score.append(clf.score(X_test, y_test))



# fig = px.line(x = range(2, 10), y = tree_score)

g = plt.plot(range(2, 15), clf_score)

g = plt.plot(clf_score.index(max(clf_score)) + min(range(2,15)), max(clf_score), 'o')

g = plt.title('RandomForestClassifier')

print(max(clf_score))
# Neural Network



from sklearn.neural_network import MLPClassifier



clf_score = []

layers = [10, 25, 50, 100, 150, 200, 250, 300, 500]



for hidden_layer_sizes in layers:

    clf = MLPClassifier(hidden_layer_sizes= hidden_layer_sizes , 

                        random_state = 15).fit(X_train, y_train)

    clf_score.append(clf.score(X_test, y_test))



g = plt.plot(layers, clf_score)

g = plt.plot(layers[clf_score.index(max(clf_score))], max(clf_score), 'o')

g = plt.title('Neural Network')

print(max(clf_score))