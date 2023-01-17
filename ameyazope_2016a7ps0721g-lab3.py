import numpy as np

import pandas as pd

import random 



random.seed(12)

pd.options.display.max_rows = 9999

pd.options.display.max_columns = 999
train = pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')

test = pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')
train.head()
(test.TotalCharges==' ').sum()
test[test['TotalCharges']==' ']
test.loc[71,'TotalCharges']=0

test.loc[580,'TotalCharges']=0

test.loc[637,'TotalCharges']=0

test.loc[790,'TotalCharges']=0

test.loc[1505,'TotalCharges']=0
train[train['TotalCharges']==' ']
train.loc[544,'TotalCharges']=0

train.loc[1348,'TotalCharges']=0

train.loc[1553,'TotalCharges']=0

train.loc[2504,'TotalCharges']=0

train.loc[3083,'TotalCharges']=0

train.loc[4766,'TotalCharges']=0
train.HighSpeed.unique()
labelEncoderColumns = ['gender','SeniorCitizen','Married','Children','Internet','AddedServices']
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

for i in labelEncoderColumns:

    train[i] = le.fit_transform(train[i])

    test[i] = le.fit_transform(test[i])
hotEncoderColumns = ['TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','HighSpeed','Subscription','PaymentMethod']
train = pd.get_dummies(train,columns=hotEncoderColumns,drop_first=True)

test = pd.get_dummies(test,columns=hotEncoderColumns,drop_first=True)

train.head()
y = train['Satisfied']

x = train.drop(['Satisfied','custId'],axis=1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x)

x = scaler.transform(x)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=999)
from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters=3, random_state=0,).fit(x_train)

y_ans = kmeans.predict(x_train)
x
#Voting



## Class Zero

zero_0=0

zero_1=0



one_0=0

one_1=0



two_0=0

two_1=0

for i in range(len(y_ans)):

    if y_ans[i]==0:

        if y[i]==0:

            zero_0+=1

        else:

            zero_1+=1

    elif y_ans[i]==1:

        if y[i]==0:

            one_0+=1

        else:

            one_1+=1

    elif y_ans[i]==2:

        if y[i]==0:

            two_0+=1

        else:

            two_1+=1

            

y_ans = kmeans.predict(x_test)

for i in range(len(y_ans)):

    if y_ans[i]==0:

        if zero_0>=zero_1:

            y_ans[i]=0

        else:

            y_ans[i]=1

    elif y_ans[i]==1:

        if one_0>=one_1:

            y_ans[i]=0

        else:

            y_ans[i]=1

    elif y_ans[i]==2:

        if two_0>=two_1:

            y_ans[i]=1

        else:

            y_ans[i]=0
np.unique(y_ans)
from sklearn.metrics import accuracy_score

accuracy_score(y_ans,y_test)
y_test.shape
test.head()
test = test.drop(['custId'],axis=1)

test = scaler.transform(test)
y_pred = kmeans.predict(test)
for i in range(len(y_pred)):

    if y_pred[i]==0:

        if zero_0>zero_1:

            y_pred[i]=0

        else:

            y_pred[i]=1

    elif y_pred[i]==1:

        if one_0>one_1:

            y_pred[i]=0

        else:

            y_pred[i]=1

    elif y_pred[i]==2:

        if two_0>two_1:

            y_pred[i]=1

        else:

            y_pred[i]=0
idq=pd.read_csv("test.csv")

idw=idq[['custId']]

data_sub=pd.DataFrame(y_pred)

data_fin=idw.join(data_sub[0],how='left')

data_fin.rename(columns= {0:'Satisfied'},inplace=True)



data_fin = data_fin.astype(int)

data_fin.head()

#for i in range(data_fin.shape[0]):

#    if data_fin.loc[i,'Satisfied']==1:

#        data_fin.loc[i,'Satisfied']=0

#    else:

#        data_fin.loc[i,'Satisfied']=1

data_fin.Satisfied.value_counts()
(data_fin['Satisfied']==0).sum()
test.shape
data_fin.to_csv('submission41.csv',columns=['custId','Satisfied'],index=False)
import numpy as np

import pandas as pd

import random 

from sklearn.cluster import KMeans



random.seed(12)

pd.options.display.max_rows = 9999

pd.options.display.max_columns = 999
train = pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')

test = pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')
train.head()
(test.TotalCharges==' ').sum()
test[test['TotalCharges']==' ']
test.loc[71,'TotalCharges']=0

test.loc[580,'TotalCharges']=0

test.loc[637,'TotalCharges']=0

test.loc[790,'TotalCharges']=0

test.loc[1505,'TotalCharges']=0
train[train['TotalCharges']==' ']
train.loc[544,'TotalCharges']=0

train.loc[1348,'TotalCharges']=0

train.loc[1553,'TotalCharges']=0

train.loc[2504,'TotalCharges']=0

train.loc[3083,'TotalCharges']=0

train.loc[4766,'TotalCharges']=0
train.HighSpeed.unique()
labelEncoderColumns = ['gender','SeniorCitizen','Married','Children','Internet','AddedServices']
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

for i in labelEncoderColumns:

    train[i] = le.fit_transform(train[i])

    test[i] = le.fit_transform(test[i])
hotEncoderColumns = ['TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','HighSpeed','Subscription','PaymentMethod']
train = pd.get_dummies(train,columns=hotEncoderColumns,drop_first=True)

test = pd.get_dummies(test,columns=hotEncoderColumns,drop_first=True)

train.head()
y = train['Satisfied']

x = train.drop(['Satisfied',],axis=1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x)

x = scaler.transform(x)

test=scaler.transform(test)
cols = list(train.columns)

cols.remove('Satisfied')

x = pd.DataFrame(x,columns  = cols)

test = pd.DataFrame(test,columns  = cols)

km = KMeans(5,random_state=666)

km2 = KMeans(10,random_state=474)

km3 = KMeans(7,random_state=5153)

x['km_5']= km.fit_predict(x)

test['km_5'] = km.fit_predict(test)

x['km_10']= km2.fit_predict(x)

test['km_10'] = km2.fit_predict(test)

x['km_7']= km3.fit_predict(x)

test['km_7']= km3.fit_predict(test)


x =  scaler.fit_transform(x)

test  = scaler.transform(test)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters=3, random_state=71).fit(x_train)

y_ans = kmeans.predict(x_train)
x
#Voting



## Class Zero

zero_0=0

zero_1=0



one_0=0

one_1=0



two_0=0

two_1=0

for i in range(len(y_ans)):

    if y_ans[i]==0:

        if y[i]==0:

            zero_0+=1

        else:

            zero_1+=1

    elif y_ans[i]==1:

        if y[i]==0:

            one_0+=1

        else:

            one_1+=1

    elif y_ans[i]==2:

        if y[i]==0:

            two_0+=1

        else:

            two_1+=1

            

y_ans = kmeans.predict(x_test)

for i in range(len(y_ans)):

    if y_ans[i]==0:

        if zero_0>=zero_1:

            y_ans[i]=0

        else:

            y_ans[i]=1

    elif y_ans[i]==1:

        if one_0>=one_1:

            y_ans[i]=0

        else:

            y_ans[i]=1

    elif y_ans[i]==2:

        if two_0>=two_1:

            y_ans[i]=1

        else:

            y_ans[i]=0
np.unique(y_ans)
from sklearn.metrics import accuracy_score

accuracy_score(y_ans,y_test)
y_test.shape
y_pred = kmeans.predict(test)
for i in range(len(y_pred)):

    if y_pred[i]==0:

        if zero_0>zero_1:

            y_pred[i]=0

        else:

            y_pred[i]=1

    elif y_pred[i]==1:

        if one_0>one_1:

            y_pred[i]=0

        else:

            y_pred[i]=1

    elif y_pred[i]==2:

        if two_0>two_1:

            y_pred[i]=1

        else:

            y_pred[i]=0
idq=pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")

idw=idq[['custId']]

data_sub=pd.DataFrame(y_pred)

data_fin=idw.join(data_sub[0],how='left')

data_fin.rename(columns= {0:'Satisfied'},inplace=True)



data_fin = data_fin.astype(int)

data_fin.head()

#for i in range(data_fin.shape[0]):

#    if data_fin.loc[i,'Satisfied']==1:

#        data_fin.loc[i,'Satisfied']=0

#    else:

#        data_fin.loc[i,'Satisfied']=1

data_fin.Satisfied.value_counts()
(data_fin['Satisfied']==0).sum()
test.shape
data_fin.to_csv('trial_submission_jai_mata_di.csv',columns=['custId','Satisfied'],index=False)