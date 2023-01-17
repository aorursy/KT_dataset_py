import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



%matplotlib inline
df=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.drop(['sl_no'],axis=1,inplace=True)
df['status'].values[df['status']=='Not Placed'] = 0 

df['status'].values[df['status']=='Placed'] = 1

df.status = df.status.astype('int')
df.head()
sns.heatmap(df.isnull(), cbar=False) #finding columns having nan values
df['salary'] = df['salary'].replace(np.nan, 0) #Replace Nan with 0
df.describe()
sns.pairplot(df,kind='reg')
plt.figure(figsize=(14,12))

sns.heatmap(df.corr(), linewidth=0.2, cmap="YlGnBu", annot=True)
sns.countplot(df['status'])
sns.countplot(df['gender']) #to see the composition of gender

plt.show()
sns.countplot(df['status'],hue=df['gender'])

plt.show()
Boys_placed=100

Total_Boys=140

Boys_placed_prop=Boys_placed/Total_Boys





Girls_placed=50

Total_Girls=70

Girls_placed_prop=Girls_placed/Total_Girls



print('Proportion of boys got placed: ') , 

print(Boys_placed_prop)



print('\nProportion of girls got placed: ') , 

print(Girls_placed_prop)

fig,axes = plt.subplots(3,2, figsize=(20,12))

sns.barplot(x='status', y='ssc_p', data=df, ax=axes[0][0])

sns.barplot(x='status', y='hsc_p', data=df, ax=axes[0][1])

sns.barplot(x='status', y='degree_p',data=df, ax=axes[1][0])

sns.barplot(x='status', y='etest_p',data=df, ax=axes[1][1])

sns.barplot(x='status', y='mba_p', data=df, ax=axes[2][0])

fig.delaxes(ax = axes[2][1]) 
sns.catplot(x="status", y="ssc_p", data=df,kind="swarm")

sns.catplot(x="status", y="hsc_p", data=df,kind="swarm")

sns.catplot(x="status", y="degree_p", data=df,kind="swarm")

sns.catplot(x="status", y="etest_p", data=df,kind="swarm")

sns.catplot(x="status", y="mba_p", data=df,kind="swarm")

plt.show()
df.groupby(['workex','status']).count()['salary']
sns.countplot(df['workex']) #to see the composition of work experience

plt.show()
sns.countplot(df['status'],hue=df['workex'])

plt.show()
Y_placed=64

Total_Y=74

Y_placed_prop=Y_placed/Total_Y





N_placed=84

Total_N=141

N_placed_prop=N_placed/Total_N



print('Proportion of student with work experience got placed: ') , 

print(Y_placed_prop)



print('\nProportion of students with No work experience got placed: ') , 

print(N_placed_prop)

df.groupby(['specialisation','status']).count()['salary']
sns.countplot(df['specialisation']) #to see the composition of work experience

plt.show()
sns.countplot(df['status'],hue=df['specialisation'])

plt.show()
MH_placed=53

Total_MH=95

MH_placed_prop=MH_placed/Total_MH





MF_placed=95

Total_MF=120

MF_placed_prop=MF_placed/Total_MF



print('Proportion of student from Market and HR got placed: ') , 

print(MH_placed_prop)



print('\nProportion of students from Market and finance got placed: ') , 

print(MF_placed_prop)

df.groupby(['ssc_b','status']).count()['salary']
sns.countplot(df['ssc_b'])

plt.show()
sns.countplot(df['status'],hue=df['ssc_b'])

plt.show()
print('Proportion of student having central board in SSC got placed: ') , 

print(78/(78+38))



print('\nProportion of students having other board in SSC got placed: ') , 

print(70/(70+29))

df.groupby(['hsc_b','status']).count()['salary']
sns.countplot(df['hsc_b'])

plt.show()
sns.countplot(df['status'],hue=df['hsc_b'])

plt.show()
print('Proportion of student having central board in HSC got placed: ') , 

print(57/(57+27))



print('\nProportion of students having other board in HSC got placed: ') , 

print(91/(91+40))

df.groupby(['hsc_s','status']).count()['salary']
sns.countplot(df['status'],hue=df['hsc_s'])

plt.show()
print('Proportion of commerce student got placed: ') , 

print(79/(79+34))



print('\nProportion of science students got placed: ') , 

print(63/(63+28))

df.groupby(['degree_t','status']).count()['salary']
sns.countplot(df['status'],hue=df['degree_t'])

plt.show()
print('Proportion of Comm&Mgmt student got placed: ') , 

print(102/(43+102))



print('\nProportion of Sci&Tech students got placed: ') , 

print(41/(41+18))

def cat_to_num(data_x,col):

    dummy = pd.get_dummies(data_x[col])

    del dummy[dummy.columns[-1]]#To avoid dummy variable trap

    data_x= pd.concat([data_x,dummy],axis =1)

    return data_x
df.columns
df_x=df[[ 'ssc_p', 'hsc_p',  'hsc_s', 'degree_p',

       'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p']]
for i in df_x.columns:

    if df_x[i].dtype ==object:

        print(i)

        df_x =cat_to_num(df_x,i)
df_x.drop(['workex','specialisation','hsc_s','degree_t'],inplace =True,axis =1)
y = df['status']

X = df_x
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)
y_predict = model.predict(X_test)



model.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=1000, random_state=42)

model.fit(X_train, y_train)
y_predict = model.predict(X_test)



model.score(X_test, y_test)