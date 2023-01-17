import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.offline as offline

import plotly.graph_objs as go

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,roc_curve,auc

from sklearn.model_selection import train_test_split





offline.init_notebook_mode(connected=True) 
credit_card = pd.read_csv("../input/creditcard.csv")

credit_card.head()
credit_card.shape #analyze the number of records
credit_card.isnull().values.any()
credit_card.describe()
class_imb=credit_card['Class'].value_counts()



data=[go.Bar(x=class_imb.keys().tolist(),y=class_imb.tolist(),marker=dict(color="Red"))]



layout = go.Layout(title = 'Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1',

                   xaxis=dict(title='Class', showticklabels=True),

                   yaxis=dict(title='Number of transactions'))

fig=go.Figure(data=data,layout=layout)

offline.iplot(fig)
normal_trans=credit_card[credit_card['Class']==0]

fraud_trans=credit_card[credit_card['Class']==1]







trace0 = go.Box(

    y=normal_trans.Amount,name = "Normal"

)

trace1 = go.Box(

    y=fraud_trans.Amount,name = "Fraud"

)

data = [trace0, trace1]

offline.iplot(data)
fraud = credit_card.loc[credit_card['Class'] == 1]



data = [go.Scatter(

    x = fraud['Time'],y = fraud['Amount'],

    name="Amount",

     marker=dict(

                color='rgb(238,23,11)',

                line=dict(

                    color='red',

                    width=1),

                opacity=0.5,

            ),

    text= fraud['Amount'],

    mode = "markers"

)]



layout = dict(title = 'Amount of fraudulent transactions',

          xaxis = dict(title = 'Time [s]', showticklabels=True), 

          yaxis = dict(title = 'Amount')

         )

fig = dict(data=data, layout=layout)

offline.iplot(fig)
credit_card.hist(figsize=(20,20))

plt.show()
columns=credit_card.columns

feature_columns=columns.delete(len(columns)-1)



X=credit_card[feature_columns] #data

y=credit_card['Class']         #label
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=2019)
balance_sampling=SMOTE(random_state=2019)

X_balance,y_balance=balance_sampling.fit_sample(X_train,y_train)

len(y_balance[y_balance==1]) #check for the increase in Fraud labels
clf=RandomForestClassifier(n_jobs=4, #number of parallel jobs

                             random_state=2019,

                             criterion='gini',

                             n_estimators=100,#number of estimators

                             verbose=False)

clf.fit(X_balance,y_balance)

prediction=clf.predict(X_test)
confusion_matrix=(y_test,prediction)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, prediction)

roc_auc = auc(false_positive_rate, true_positive_rate)

print (roc_auc)