#importing python libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py 
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import os
#reading data
df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
df.info() #There isn't any NaN values
#I split dataframe by customers left or stayed.
dfy = df[df.Churn == 'Yes'] #dfy : dataframe churn : yes

dfn = df[df.Churn == 'No'] #dfn : dataframe churn : no

trace0 = go.Histogram(
    x = dfy.TotalCharges,
    name = 'Left',
    marker=dict(color='rgba(255, 0, 0, 0.5)'))

trace1 = go.Histogram(
    x = dfn.TotalCharges,
    name = 'Stayed',
    marker=dict(color='rgba(0, 153, 255, 0.5)'))

layout = go.Layout(title = 'Total Charges of Customers',
                   xaxis=dict(title='Total Charges'),
                   yaxis=dict( title='Count'),
                   )
data = [trace0,trace1]

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace0 = go.Histogram(
    x = dfy.MonthlyCharges,
    name = 'Left',
    marker=dict(color='rgba(255, 0, 0, 0.5)'))

trace1 = go.Histogram(
    x = dfn.MonthlyCharges,
    name = 'Stayed',
    marker=dict(color='rgba(0, 153, 255, 0.5)'))


layout = go.Layout(title = 'Monthly Charges of Customers',
                   xaxis=dict(title='Monthly Charges'),
                   yaxis=dict( title='Count'),
                   )
data = [trace0,trace1]

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace0 = go.Histogram(
    x = dfy.tenure,
    name = 'Left',
    marker=dict(color='rgba(255, 0, 0, 0.5)'))

trace1 = go.Histogram(
    x = dfn.tenure,
    name = 'Stayed',
    marker=dict(color='rgba(0, 153, 255, 0.5)'))



layout = go.Layout(title = 'Tenure of Customers',
                   xaxis=dict(title='Tenure'),
                   yaxis=dict( title='Count'),
                   )
data = [trace0,trace1]

fig = go.Figure(data=data, layout=layout)

iplot(fig)
plt.figure(figsize = (15,15))
plt.suptitle('Statistics of Left Customers',fontsize = 16)

plt.subplot(2,2,1)
sns.countplot(dfy.gender)


plt.subplot(2,2,2)
sns.countplot(dfy.SeniorCitizen)

plt.subplot(2,2,3)
sns.countplot(dfy.Partner)

plt.subplot(2,2,4)
sns.countplot(dfy.Dependents)
plt.show()
plt.figure(figsize = (15,15))
plt.suptitle('Statistics of Stayed Customers',fontsize = 16)

plt.subplot(2,2,1)
sns.countplot(dfn.gender)


plt.subplot(2,2,2)
sns.countplot(dfn.SeniorCitizen)

plt.subplot(2,2,3)
sns.countplot(dfn.Partner)

plt.subplot(2,2,4)
sns.countplot(dfn.Dependents)
plt.show()
plt.figure(figsize = (15,15))
plt.suptitle('Statistics of Left Customers',fontsize = 16)

plt.subplot(4,3,1)
sns.countplot(dfy.PhoneService)

plt.subplot(4,3,2)
sns.countplot(dfy.MultipleLines)

plt.subplot(4,3,3)
sns.countplot(dfy.InternetService)

plt.subplot(4,3,4)
sns.countplot(dfy.OnlineSecurity)

plt.subplot(4,3,5)
sns.countplot(dfy.OnlineBackup)
            
plt.subplot(4,3,6)
sns.countplot(dfy.DeviceProtection)

plt.subplot(4,3,7)
sns.countplot(dfy.TechSupport)

plt.subplot(4,3,8)
sns.countplot(dfy.StreamingTV)

plt.subplot(4,3,9)
sns.countplot(dfy.StreamingMovies)

plt.subplot(4,3,10)
sns.countplot(dfy.Contract)

plt.subplot(4,3,11)
sns.countplot(dfy.PaperlessBilling)
plt.show()
plt.figure(figsize = (15,15))
plt.suptitle('Statistics of Stayed Customers',fontsize = 16)

plt.subplot(4,3,1)
sns.countplot(dfn.PhoneService)

plt.subplot(4,3,2)
sns.countplot(dfn.MultipleLines)

plt.subplot(4,3,3)
sns.countplot(dfn.InternetService)

plt.subplot(4,3,4)
sns.countplot(dfn.OnlineSecurity)

plt.subplot(4,3,5)
sns.countplot(dfn.OnlineBackup)
            
plt.subplot(4,3,6)
sns.countplot(dfn.DeviceProtection)

plt.subplot(4,3,7)
sns.countplot(dfn.TechSupport)

plt.subplot(4,3,8)
sns.countplot(dfn.StreamingTV)

plt.subplot(4,3,9)
sns.countplot(dfn.StreamingMovies)

plt.subplot(4,3,10)
sns.countplot(dfn.Contract)

plt.subplot(4,3,11)
sns.countplot(dfn.PaperlessBilling)
plt.show()
df.drop(labels = 'customerID',axis = 1,inplace = True)#i dont need this feature
#Make features numeric
df.gender = [1 if i == 'Male' else 0 for i in df.gender]

df.InternetService = [2 if i == 'Fiber optic' else 1 if i == 'DSL' else 0 for i in df.InternetService]

df.Contract = [2 if i == 'Two year' else 1 if i == 'One year' else 0 for i in df.Contract]

df.PaymentMethod = [2 if i == 'Electronic check' else 1 if i == 'Mailed check' else 0 for i in df.PaymentMethod]

#I will create a for loop because i will do similar processes
l = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
     'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']

for i in l:
    df[i] = [1 if j == 'Yes' else 0 for j in df[i]]

#transforming datatype to float

df.TotalCharges.replace(' ','0.0',inplace = True)#if i dont replace spaces i can't make values float

df.TotalCharges = df.TotalCharges.astype(float)
df.info()
#normalization
df = (df - np.min(df))/(np.max(df) - np.min(df))
df.head()
dfx = df.drop(['Churn'],axis = 1)

dfy = df.loc[:,['Churn']]
from sklearn.model_selection import train_test_split #spliting data to train data and test data

x_train, x_test, y_train, y_test = train_test_split(dfx,dfy,test_size = 0.2,random_state=1)

from sklearn.ensemble import RandomForestClassifier

score_list = [] #to determine best number of estimators parameter for Random Forest

for i in range(1,100):
    rf = RandomForestClassifier(n_estimators = i,random_state = 40)
    rf.fit(x_train,y_train.values.ravel())
    score_list.append(rf.score(x_test,y_test))

plt.figure(figsize = (15,9))
plt.xlabel('Prediction Score')
plt.ylabel('Number of Parameters')
plt.plot(range(1,100),score_list)
plt.show()

print('Best parameter is',np.argmax(score_list))
rf = RandomForestClassifier(n_estimators = 79,random_state = 40)

rf.fit(x_train,y_train.values.ravel())

print('Final prediction score of algorithm is',rf.score(x_test,y_test))