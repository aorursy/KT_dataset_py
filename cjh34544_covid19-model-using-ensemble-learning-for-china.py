import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv

from matplotlib import pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')
train.head(5)
import seaborn as sns
train.info()
df1 = train.Population.groupby(train['Country_Region']).max().sort_values(ascending= False)

df10 = pd.DataFrame()

df20 = pd.DataFrame()

df30 = pd.DataFrame()

df40 = pd.DataFrame()

df10['population'] = df1.iloc[0:10]

df10['country']= df10.index

df20['population'] = df1.iloc[11:20]

df20['country'] = df20.index

df30['population'] = df1.iloc[21:30]

df30['country'] = df30.index

df40['population'] = df1.iloc[31:40]

df40['country'] = df40.index
plt.figure(figsize =(10,10))

plt.subplot(2,1,1)

sns.barplot(x='country', y='population', data=df10, orient ='v')

plt.xlabel('Country')

plt.title('Popoulation Top 10')

plt.subplot(2,1,2)

sns.barplot(x='country', y='population', data=df40, orient ='v')

plt.xlabel('Country')

plt.title('Population Last 10')
train1= train[train['Target']=='ConfirmedCases']

data1 = pd.DataFrame()

data1['values'] =train1.TargetValue.groupby(train1['Country_Region']).sum().sort_values(ascending= False)

data1['country'] = data1.index

data1.index = np.arange(0,len(data1))

data10 = data1.iloc[0:10,:]

data20 = data1.iloc[11:20,:]
plt.figure(figsize =(10,10))

plt.subplot(2,1,1)

sns.barplot(x='country', y='values', data=data10, orient ='v')

plt.xlabel('Country')

plt.ylabel('Cases')

plt.title('Covid Cases Top 10')

plt.subplot(2,1,2)

sns.barplot(x='country', y='values', data=data20, orient ='v')

plt.xlabel('Country')

plt.ylabel('Cases')

plt.title('Covid Cases Next 10')
train1= train[train['Target']!='ConfirmedCases']

data1 = pd.DataFrame()

data1['values'] =train1.TargetValue.groupby(train1['Country_Region']).sum().sort_values(ascending= False)

data1['country'] = data1.index

data1.index = np.arange(0,len(data1))

data10 = data1.iloc[0:10,:]

data20 = data1.iloc[11:20,:]
plt.figure(figsize =(10,10))

plt.subplot(2,1,1)

sns.barplot(x='country', y='values', data=data10, orient ='v')

plt.xlabel('Country')

plt.ylabel('Deaths')

plt.title('Covid Cases Top 10')

plt.subplot(2,1,2)

sns.barplot(x='country', y='values', data=data20, orient ='v')

plt.xlabel('Country')

plt.ylabel('Deaths')

plt.title('Covid Cases Next 10')
china = train[train['Country_Region']=='China']
china.info()
china.drop(['County','Province_State'],axis =1,inplace =True)
china.head(5)
china.index = np.arange(0,len(china)) #rechanging the index
china.head(5)
chi = china[china['Target']=='ConfirmedCases']

chi.index = np.arange(0,len(chi))
chi.head(5)
print("Date with more no.of cases in china {}".format((chi[chi['TargetValue']==chi['TargetValue'].max()]['Date']).values))

print("The cases are {}".format((chi[chi['TargetValue']==chi['TargetValue'].max()]['TargetValue']).values))
list1 = []

for i in range(2,7):

    date = '2020'+'-0'+str(i)+'-01'

    list1.append(chi[chi['Date']<date]['TargetValue'].sum())

print(list1)
sns.barplot(['upto Jan','Upto Feb','Upto Mar', 'Upto Apr','Upto May'],list1)
list2 =[]

for i in range(len(list1)):

    if i ==0:

        list2.append(list1[i])

    else:

        list2.append(list1[i]-list1[i-1])

print(list2)
labels =['Jan','Feb','Mar','Apr','May']

sns.barplot(labels,list2)
df = china['TargetValue'].groupby(train['Target']).sum()

df
labels =[df.index[0],df.index[1]]

sizes = [df[0],df[1]]

explode = (0, 0.2)  

plt.figure(figsize = (5,5))



plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)



plt.show()
wor = train[train['Target']=='ConfirmedCases']
print("Date with more no.of cases  {}".format((wor[wor['TargetValue']==wor['TargetValue'].max()]['Date']).values))

print("The cases are {}".format((wor[wor['TargetValue']==wor['TargetValue'].max()]['TargetValue']).values))

print("The Country is {}".format((wor[wor['TargetValue']==wor['TargetValue'].max()]['Country_Region']).values))
wor.columns
independent_columns = ['Country_Region','Weight','Target','Date']

dependent_column = ['TargetValue']
X= train[independent_columns]

y = train[dependent_column]
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X['Target']=le.fit_transform(X['Target'])
X.info()
train1= train[train['Target']=='ConfirmedCases']

data1 = pd.DataFrame()

data1['values'] =train1.TargetValue.groupby(train1['Country_Region']).sum().sort_values(ascending= False)

data1['country'] = data1.index
k = len(data1['country'])

dict1 = {}

for i in data1['country']:

    dict1[i] = k

    k =k-1
list1=[]

X['encoded_country']=0

for i in X['Country_Region']:

    list1.append(dict1[i])

X['encoded_country'] = list1
X['encoded_country'].value_counts()
X['date_dup'] = pd.to_datetime(X['Date'])
X['month'] = 0

list1=[]

for i in X['date_dup']:

    list1.append(i.month)

X['month'] = list1
X['date'] = 0

list1=[]

for i in X['date_dup']:

    list1.append(i.day)

X['date'] = list1
X.head(5)
X.drop(['Country_Region','Date','date_dup'],axis =1,inplace =True)
X.head(5)
plt.figure(figsize =(10,10))

sns.heatmap(X.corr(),annot=True)
from sklearn.model_selection import train_test_split as tts
max_range =10
from sklearn.ensemble import RandomForestRegressor as regr

from sklearn.metrics import r2_score
'''for i in range(max_range):

    X_train,X_test,y_train,y_test = tts(X,y,test_size =0.3,random_state =i)

    model = regr()

    model.fit(X_train,y_train)

    print("Random state {}\n".format(i))

    print(r2_score(y_test,model.predict(X_test)))'''
X_train,X_test,y_train,y_test = tts(X,y,test_size =0.3,random_state =7)

model = regr()

model.fit(X_train,y_train)
print(r2_score(y_test,model.predict(X_test)))
test = test[independent_columns]
list1=[]

test['encoded_country']=0

for i in test['Country_Region']:

    list1.append(dict1[i])

test['encoded_country'] = list1
test['date_dup'] = pd.to_datetime(test['Date'])
test['month'] = 0

list1=[]

for i in test['date_dup']:

    list1.append(i.month)

test['month'] = list1
test['date'] = 0

list1=[]

for i in test['date_dup']:

    list1.append(i.day)

test['date'] = list1
test.head(5)
test.drop(['Country_Region','Date','date_dup'],axis =1,inplace =True)
test.head(5)
le1 =LabelEncoder()

test['Target'] = le1.fit_transform(test['Target'])
pred = model.predict(test)
t =pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')

ss = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/submission.csv')
output = pd.DataFrame({'Id': t.ForecastId  , 'TargetValue': pred})
a=output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()

b=output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()

c=output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()
a.columns=['Id','q0.05']

b.columns=['Id','q0.5']

c.columns=['Id','q0.95']

a=pd.concat([a,b['q0.5'],c['q0.95']],1)

a['q0.05']=a['q0.05']

a['q0.5']=a['q0.5']

a['q0.95']=a['q0.95']
sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

sub.head()
sub.info()