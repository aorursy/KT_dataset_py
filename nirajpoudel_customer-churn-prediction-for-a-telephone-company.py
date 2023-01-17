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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.express as px

import plotly.offline as po
customer_churn = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
customer_churn.head()
print('The names of columns in our dataset:\n ',list(customer_churn.columns))

print("\nThe total number of columns in our data set: ",len(customer_churn.columns))

print("\nThe shape of our data set is: ",customer_churn.shape)
print('-------------------------------')

print("Rows\t\tMissing values")

print("-------------------------------")

print(customer_churn.isna().sum())

print('------------------------------')
print('----------------------------')

print("Rows\t\tData types")

print('----------------------------')

print(customer_churn.dtypes)

print('----------------------------')
customer_churn['Churn'][:5]
customer_churn['Churn'] = customer_churn['Churn'].replace({"Yes":1,"No":0})
customer_churn['Churn'][:5]
cols = ['OnlineBackup', 'StreamingMovies','DeviceProtection','TechSupport','OnlineSecurity','StreamingTV']

for values in cols:

    customer_churn[values] = customer_churn[values].replace({'No internet service':'No'})
customer_churn['TotalCharges'] = customer_churn['TotalCharges'].replace(" ",np.nan)



# Drop null values of 'Total Charges' feature

customer_churn = customer_churn[customer_churn["TotalCharges"].notnull()]

customer_churn = customer_churn.reset_index()[customer_churn.columns]



customer_churn['TotalCharges'] = customer_churn['TotalCharges'].astype(float)
customer_churn['TotalCharges'].dtype
customer_churn['Churn'].value_counts().unique()
churn_x = customer_churn['Churn'].value_counts().keys().tolist()

churn_y = customer_churn['Churn'].value_counts().values.tolist()

fig = px.pie(customer_churn,

           labels = churn_x,

            values = churn_y,

            color_discrete_sequence=['grey','teal'],

             hole=0.6

            )





fig.update_layout(

    title='Customer Churn',

    template='plotly_dark'

)

fig.show()
gender_chunk = customer_churn.groupby('gender').Churn.mean().reset_index()



fig = go.Figure(data=[go.Bar(

            x=gender_chunk['gender'], y=gender_chunk['Churn'],

            textposition='auto',

            width=[0.2,0.2],

            marker = dict(color=['brown','purple']))])

fig.update_layout(

    title='Churn rate by Gender',

    xaxis_title="Gender",

    yaxis_title="Churn rate",

        template='plotly_dark'



)

fig.show()
tech_chunk = customer_churn.groupby('TechSupport').Churn.mean().reset_index()



fig = go.Figure(data=[go.Bar(

            x=tech_chunk['TechSupport'], y=tech_chunk['Churn'],

            textposition='auto',

            width=[0.2,0.2],

            marker = dict(color=['midnightblue','darkgreen']))])

fig.update_layout(

    title='Churn rate by Tech Support',

    xaxis_title="Tech Support",

    yaxis_title="Churn rate",

        template='plotly_dark'



)

fig.show()
internet_chunk = customer_churn.groupby('InternetService').Churn.mean().reset_index()



fig = go.Figure(data=[go.Bar(

            x=internet_chunk['InternetService'], y=internet_chunk['Churn'],

            textposition='auto',

            width=[0.2,0.2,0.2],

            marker = dict(color=['tomato','tan','cyan']))])

fig.update_layout(

    title='Churn rate by Internet Services',

    xaxis_title="Internet Services",

    yaxis_title="Churn rate",

        template='plotly_dark'



)

fig.show()
payment_chunk = customer_churn.groupby('PaymentMethod').Churn.mean().reset_index()



fig = go.Figure(data=[go.Bar(

            x=payment_chunk['PaymentMethod'], y=payment_chunk['Churn'],

            textposition='auto',

            width=[0.2,0.2,0.2,0.2],

            marker = dict(color=['teal','thistle','lime','navy']))])

fig.update_layout(

    title='Churn rate by Payment Method',

    xaxis_title="Payment Method Churns",

    yaxis_title="Churn rate",

        template='plotly_dark'



)

fig.show()
contract_chunk = customer_churn.groupby('Contract').Churn.mean().reset_index()



fig = go.Figure(data=[go.Bar(

            x=contract_chunk['Contract'], y=contract_chunk['Churn'],

            textposition='auto',

            width=[0.2,0.2,0.2,0.2],

            marker = dict(color=['teal','thistle','purple']))])

fig.update_layout(

    title='Churn rate by Contract',

    xaxis_title="Contract Churns",

    yaxis_title="Churn rate",

        template='plotly_dark'



)

fig.show()
ten_chunk = customer_churn.groupby('tenure').Churn.mean().reset_index()



fig = go.Figure(data=[go.Scatter(

    x=ten_chunk['tenure'],

    y=ten_chunk['Churn'],

        mode='markers',

        name='Low',

        marker= dict(size= 5,

            line= dict(width=0.8),

            color= 'blue'

           ),

)])

fig.update_layout(

    title='Churn rate by Tenure',

    xaxis_title="Tenure",

    yaxis_title="Churn rate",

    template='plotly_dark'



)

fig.show()
churn_data = pd.get_dummies(customer_churn, columns = ['Contract','Dependents','DeviceProtection','gender',

                                                        'InternetService','MultipleLines','OnlineBackup',

                                                        'OnlineSecurity','PaperlessBilling','Partner',

                                                        'PaymentMethod','PhoneService','SeniorCitizen',

                                                        'StreamingMovies','StreamingTV','TechSupport'],

                              drop_first=True)
churn_data.head()
from sklearn.preprocessing import StandardScaler



#Perform Feature Scaling on 'tenure', 'MonthlyCharges', 'TotalCharges' in order to bring them on same scale

standard = StandardScaler()

columns_for_ft_scaling = ['tenure', 'MonthlyCharges', 'TotalCharges']



#Apply the feature scaling operation on dataset using fit_transform() method

churn_data[columns_for_ft_scaling] = standard.fit_transform(churn_data[columns_for_ft_scaling])

churn_data.head()
list(churn_data.columns)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score,plot_confusion_matrix
X = churn_data.drop(['Churn','customerID'], axis=1)

y = churn_data['Churn']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
X_train.shape,y_train.shape,X_test.shape,y_test.shape
#using logistic regression

log = LogisticRegression()



#fitting our data

log.fit(X_train,y_train)



#making prediction

y_pred = log.predict(X_test)
y_pred
print("The accuracy score of Logistic Regression is: {:.2f}% ".format(accuracy_score(y_pred,y_test)*100))
#using random forest classifier

rand = RandomForestClassifier()



#fitting the data

rand.fit(X_train,y_train)



#predicting values

y_rand_pred = rand.predict(X_test)
y_rand_pred
print("The accuracy score of Random Forest is: {:.2f}% ".format(accuracy_score(y_rand_pred,y_test)*100))
#using support vector machine

svm_model = SVC(kernel='linear',probability=True)



#fitting our data

svm_model.fit(X_train,y_train)



#predicting values

y_svm_pred = svm_model.predict(X_test)

#svm_model.score(X_test,y_test)
y_svm_pred
print("The accuracy score of Support Vector Machine is: {:.2f}% ".format(accuracy_score(y_svm_pred,y_test)*100))
#using decision tree classifier

dec = DecisionTreeClassifier()



#fitting our data

dec.fit(X_train,y_train)



#predicting the values

y_dec_pred = dec.predict(X_test)
print("The accuracy score of Decision Tree is: {:.2f}% ".format(accuracy_score(y_dec_pred,y_test)*100))
#using knearestneighbor

knn = KNeighborsClassifier()



#fitting our data

knn.fit(X_train,y_train)



#predicting the values.

y_knn_pred = knn.predict(X_test)
y_knn_pred
print("The accuracy score of K-Nearest Neighbor is: {:.2f}% ".format(accuracy_score(y_knn_pred,y_test)*100))
print('Logistic Regression Model')

plot_confusion_matrix(log,X_test,y_test);
print("Support Vector Machine Model")

plot_confusion_matrix(svm_model,X_test,y_test,cmap='summer');
print("Random Forest Model")

plot_confusion_matrix(rand,X_test,y_test,cmap='cividis');
print("K-Nearest Neighbor Model")

plot_confusion_matrix(knn,X_test,y_test,cmap='magma');
print("Decision Tree Model")

plot_confusion_matrix(dec,X_test,y_test,cmap='RdPu');
# Predict the probability of Churn of each customer

churn_data['Customer Churning Probability'] = log.predict_proba(churn_data[X_test.columns])[:,1]
churn_data[['customerID','Customer Churning Probability']].head(10)