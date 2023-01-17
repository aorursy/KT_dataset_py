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
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns



%matplotlib inline
churn=pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
churn.columns.values
churn.dtypes
# change totalcharges's type(object) to float64
churn['TotalCharges']=pd.to_numeric(churn['TotalCharges'],errors = 'coerce')
churn.isnull().sum()
churn.dropna(inplace=True)
# delete the first column (customerID)

churn.head()

churn_d=churn.iloc[:,1:]
# transfrom yes,no to 1,0

churn_d['Churn'].replace(to_replace='Yes',value=1,inplace=True) 

churn_d['Churn'].replace(to_replace='No',value=0,inplace=True)
#get_dummy 

churn_dum= pd.get_dummies(churn_d)

churn_dum.columns.values
sns.countplot(x="Churn",data=churn)
plt.figure(figsize=(15,8)) 

churn_corr = churn_dum.corr()['Churn'].sort_values(ascending=False).plot(kind='bar', 

                                                                         title ='Correlation between Churn & variables'

                                                                        )

churn_corr.set_xlabel('category',fontsize=20) 

churn_corr.set_ylabel('correlation',fontsize=20)
plt.figure(figsize=(10,8), dpi= 80) 

sns.heatmap(churn.corr(), xticklabels=churn.corr().columns, 

            yticklabels=churn.corr().columns, cmap='RdYlGn', center=0, annot=True)
# more exploration 
for item in churn['Contract'].unique():

    print(item)
contract_types = (churn['Contract'].value_counts(normalize=True) * 100).keys().tolist() 

contract_propotion = (churn['Contract'].value_counts(normalize=True) * 100).values.tolist()

for i in range(3):

    contract_propotion[i]=round(contract_propotion[i],2)

text= ['{} %'.format(x) for x in contract_propotion]

print(contract_types)
month_to_month =churn.loc[churn['Contract']=='Month-to-month'] 

m2m = int(round((month_to_month['Churn'].value_counts(normalize=True) * 100)['Yes']))

one_year =churn.loc[churn['Contract']=='One year'] 

oney = int(round((one_year['Churn'].value_counts(normalize=True) * 100)['Yes']))

two_year =churn.loc[churn['Contract']=='Two year'] 

twoy = int(round((two_year['Churn'].value_counts(normalize=True) * 100)['Yes']))

churn_rate = [m2m, oney, twoy] 

retention_rate = [100 - m2m, 100 - oney, 100 - twoy]

print(contract_types,'\n',contract_propotion,'\n',churn_rate,'\n',retention_rate)
#Visulize the result above 



plt.figure(figsize=(8,6)) 

churn_label=['{} %'.format(x) for x in churn_rate] 

retention_label=['{} %'.format(x) for x in retention_rate]

p1=plt.bar(contract_types,churn_rate,color='yellow', hatch="*") 

p2=plt.bar(contract_types, retention_rate,bottom=churn_rate,color='#FFE4C4')

plt.ylim(0,100) 

plt.ylabel('churn and retention rate') 

plt.xlabel('contract type')
import plotly.express as px 

fig = px.histogram(churn, x="Churn", y="MonthlyCharges", color='Churn', facet_col="Contract", histfunc='avg')

fig.update_layout(title_text='Average Monthly Cost by Contract Type')

fig.show()
#Tenure: Number of months the customer has stayed with the company

churn["tenure"].describe()
fig, a1=plt.subplots(nrows=1,ncols=1,sharey=True,figsize=(8,4))

for type in contract_types:

    for ai in [a1]:

        for title in ['M-to-M','One-year','Two-year']:

            a=sns.distplot(churn[churn['Contract']==type]['tenure'],ax=ai)

            a.set_xlabel('Tenure_month')

            a.set_ylabel('proportion of customer') 

            a.set_title('Churn & Tenure')

fig.legend(labels=['M-to-M','one-year','two-year'])
# build a list of service including all...



service=['PhoneService','MultipleLines','InternetService','OnlineSecurity', 

         'OnlineBackup','DeviceProtection', 'TechSupport','StreamingTV','StreamingMovies']
fig,axes=plt.subplots(nrows=3,ncols=3,figsize=(12,10)) 



for i,item in enumerate(service):

    if i <3:

        ax=churn[item].value_counts().plot(kind='bar',ax=axes[i,0]) 

    if i >= 3 and i < 6:

        ax=churn[item].value_counts().plot(kind='bar',ax=axes[i-3,1]) 

    if i>=6:

        ax=churn[item].value_counts().plot(kind='bar',ax=axes[i-6,2]) 

    ax.set_title(item)
churn_dum.head() # get_dummy before
# choose variables 

x=churn_dum.drop(columns=['Churn']) 

y=churn_dum['Churn'] 

#normalization 

from sklearn.preprocessing import MinMaxScaler 

features=x.columns.values 

scaler=MinMaxScaler(feature_range=(0,1)) 

scaler.fit(x) 

x=pd.DataFrame(scaler.transform(x)) 

x.columns=features
from sklearn.model_selection import train_test_split 

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression 

model=LogisticRegression() 

result=model.fit(x_train,y_train)



from sklearn import metrics

lg_pred=model.predict(x_test)

print(metrics.accuracy_score(y_test,lg_pred))
from sklearn.metrics import classification_report 

lg_report= classification_report(y_test, lg_pred) 

print(lg_report)
from sklearn.ensemble import RandomForestClassifier 

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=101) 

model_r=RandomForestClassifier(n_estimators=100,random_state=50, oob_score=True) 

model_r.fit(x_train,y_train) 

rf_pred=model_r.predict(x_test)

print(metrics.accuracy_score(y_test,rf_pred))
rf_report=classification_report(y_test, rf_pred)

print(rf_report)
rand_score =[] 

for i in range(1,102):

    i_loop= RandomForestClassifier(n_estimators=i, random_state=101) 

    i_loop.fit(x_train,y_train) 

    loop_pre=i_loop.predict(x_test) 

    rand_score.append(metrics.accuracy_score(y_test,loop_pre))
plt.plot(range(1,102),rand_score)

plt.xlabel("Range") 

plt.ylabel("accuracy score")

plt.show()
model_r=RandomForestClassifier(n_estimators=79,random_state=101, oob_score=True)

model_r.fit(x_train,y_train) 

rf_new_pred=model_r.predict(x_test)

print(metrics.accuracy_score(y_test,rf_new_pred))
# only a little bit improvement 
rf_new_report=classification_report(y_test, rf_new_pred)

print(rf_new_report)
rand_coef=model_r.feature_importances_ 

weight=pd.Series(rand_coef,index=x.columns.values)

weight.sort_values(ascending=False)[:10]
from sklearn.svm import SVC

svc_model = SVC(random_state=101)

svc_model.fit(x_train, y_train) 

accuracy_svc = svc_model.score(x_test, y_test)

print(accuracy_svc)
svm_pred=svc_model.predict(x_test)



from sklearn.metrics import classification_report, confusion_matrix

svm_report=classification_report(y_test,svm_pred) 

print(svm_report)
x_train.head() # has been standardalized
import keras
from keras.models import Sequential 

from keras.layers import Dense 

from keras import optimizers 

model_nn = Sequential() 

sgd= optimizers.SGD(lr=0.01) # (set learning rate)

# first time, no gradient descend, so might be the reason learning rate is too fast, thus...set it as 0.01
model_nn.add(Dense(45, activation= 'relu', input_dim=45)) 

model_nn.add(Dense(22, activation='relu'))

model_nn.add(Dense(11, activation='relu')) 

model_nn.add(Dense(6, activation='relu')) 

model_nn.add(Dense(1,activation='sigmoid'))
model_nn.compile(loss='binary_crossentropy',optimizer='SGD',metrics=['accuracy']) 

history= model_nn.fit(x_train, y_train,

                      batch_size=30, epochs=50, validation_data=(x_test,y_test))
result_1=model_nn.evaluate(x_test,y_test)

result_1
history_dict=history.history

history_dict.keys()
loss_value=history_dict['loss'] 

val_loss_values= history_dict['val_loss']
plt.clf() 

epochs=range(1,len(loss_value)+1)

plt.plot(epochs,loss_value,'bo',label='Training loss') 

plt.plot(epochs,val_loss_values,'b',label='Validation loss') 

plt.title('Training and validation loss') 

plt.xlabel('Epochs') 

plt.ylabel('loss')

plt.legend()
 # Accroding to the graph, it is reasonable to use epochs=50 where has the least loss 
nn_pred = model_nn.predict(x_test)

nn_pred[1:5]
def threshold(nn_pred):

    lst_threshold=[] 

    for i in nn_pred:

        if i >=0.5:

            i=1 

            lst_threshold.append(i)

        else: 

            i=0 

            lst_threshold.append(i)

    return lst_threshold
nn_pred_new = threshold(nn_pred) 

nn_pred_new[1:5]
nn_report=classification_report(y_test,nn_pred_new)

print(nn_report)
print('\n--------------------Logistic Regression-----------------\n',lg_report, 

      '\n----------------------Random Forest---------------------\n',rf_report, 

      '\n----------------- Support Vector Machine----------------\n',svm_report, 

      '\n---------------------Neural Network---------------------\n',nn_report)