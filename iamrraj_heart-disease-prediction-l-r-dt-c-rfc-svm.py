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
df = pd.read_csv("../input/heart.csv")

x = df.iloc[:,0:13].values

y =  df.iloc[:,13].values
x
#Spliting the dataset into training and tst set

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state = 0)
sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)

from sklearn.linear_model import LogisticRegression

classfier = LogisticRegression(random_state = 0)

classfier.fit(x_train,y_train)
y_pred = classfier.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score

cm = confusion_matrix(y_test,y_pred)

accuracy = accuracy_score(y_test,y_pred)

print("Logistic Regression :")

print("Accuracy = ", accuracy)

print(cm)
df.info
df.head()
df.describe()
df.size
df.shape
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set_style(style="whitegrid")
print(df.sex.value_counts())

plt.figure(figsize=(15,7))

sns.countplot('sex', data=df)

df.columns
sns.distplot(df['age'], color='green')
# sns.pairplot(df, hue='age')
sns.barplot('sex','chol', data= df)
df.rename(columns={'chol':'cholestrol'})[:5]
df = pd.read_csv("../input/heart.csv")
plt.figure(figsize=(15,6))

sns.barplot('age','chol', data= df)
df.groupby(['age','chol']).plot(kind='bar',stacked=True)
plt.figure(figsize=(20,10))

df.groupby(['age','sex'])['chol'].size().unstack().plot(kind='bar',figsize=(15, 6),stacked=True)

plt.show()
plt.figure(figsize=(20,10))

df.groupby(['age','sex'])['chol'].size().unstack().plot(kind='bar',figsize=(15, 6),stacked=True)

plt.show()
plt.figure(figsize=(20,10))

df.groupby(['age','sex'])['thalach'].size().unstack().plot(kind='bar',figsize=(15, 6),stacked=True)

plt.show()

# sns.barplot('age','thalach',data=df)
import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from datetime import datetime



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Imputer

from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc

import os

import warnings

warnings.filterwarnings('ignore')

print(os.listdir("../input"))
data=pd.read_csv('../input/heart.csv')
data.info()
data=data.rename(columns={'age':'Age','sex':'Sex','cp':'Cp','trestbps':'Trestbps','chol':'Chol','fbs':'Fbs','restecg':'Restecg','thalach':'Thalach','exang':'Exang','oldpeak':'Oldpeak','slope':'Slope','ca':'Ca','thal':'Thal','target':'Target'})
#Now,I will check null on all data and If data has null, I will sum of null data's. In this way, how many missing data is in the data.

print('Data Sum of Null Values \n')

data.isnull().sum()
#We will perform analysis on the training data. The relationship between the features found in the training data is observed. In this way, comments about the properties can be made



pd.plotting.scatter_matrix(data.loc[:,data.columns!='Target'],

                          c=['green','blue','red'],

                          figsize=[15,15],

                          diagonal='hist',

                          alpha=0.8,

                          s=200,

                          marker='*',

                          edgecolor='black')

plt.show()
sns.barplot(x=data.Age.value_counts()[:10].index,y=data.Age.value_counts()[:10].values)

plt.xlabel('Age')

plt.ylabel('Age Counter')

plt.title('Age Analysis System')

plt.show()
#firstly find min and max ages

minAge=min(data.Age)

maxAge=max(data.Age)

meanAge=data.Age.mean()

print('Min Age :',minAge)

print('Max Age :',maxAge)

print('Mean Age :',meanAge)
young_ages=data[(data.Age>=29)&(data.Age<40)]

middle_ages=data[(data.Age>=40)&(data.Age<55)]

elderly_ages=data[(data.Age>55)]

print('Young Ages :',len(young_ages))

print('Middle Ages :',len(middle_ages))

print('Elderly Ages :',len(elderly_ages))
sns.barplot(x=['young ages','middle ages','elderly ages'],y=[len(young_ages),len(middle_ages),len(elderly_ages)])

plt.xlabel('Age Range')

plt.ylabel('Age Counts')

plt.title('Ages State in Dataset')

plt.show()
data['AgeRange']=0

youngAge_index=data[(data.Age>=29)&(data.Age<40)].index

middleAge_index=data[(data.Age>=40)&(data.Age<55)].index

elderlyAge_index=data[(data.Age>55)].index
for index in elderlyAge_index:

    data.loc[index,'AgeRange']=2

    

for index in middleAge_index:

    data.loc[index,'AgeRange']=1



for index in youngAge_index:

    data.loc[index,'AgeRange']=0
total_genders_count=len(data.Sex)

male_count=len(data[data['Sex']==1])

female_count=len(data[data['Sex']==0])

print('Total Genders :',total_genders_count)

print('Male Count    :',male_count)

print('Female Count  :',female_count)
#Percentage ratios

print("Male State: {:.2f}%".format((male_count / (total_genders_count)*100)))

print("Female State: {:.2f}%".format((female_count / (total_genders_count)*100)))
#Male State & target 1 & 0

male_andtarget_on=len(data[(data.Sex==1)&(data['Target']==1)])

male_andtarget_off=len(data[(data.Sex==1)&(data['Target']==0)])

####

sns.barplot(x=['Male Target On','Male Target Off'],y=[male_andtarget_on,male_andtarget_off])

plt.xlabel('Male and Target State')

plt.ylabel('Count')

plt.title('State of the Gender')

plt.show()
#Female State & target 1 & 0

female_andtarget_on=len(data[(data.Sex==0)&(data['Target']==1)])

female_andtarget_off=len(data[(data.Sex==0)&(data['Target']==0)])

####

sns.barplot(x=['Female Target On','Female Target Off'],y=[female_andtarget_on,female_andtarget_off])

plt.xlabel('Female and Target State')

plt.ylabel('Count')

plt.title('State of the Gender')

plt.show()
#As seen, there are 4 types of chest pain.

data.Cp.value_counts()
sns.countplot(data.Cp)

plt.xlabel('Chest Type')

plt.ylabel('Count')

plt.title('Chest Type vs Count State')

plt.show()

#0 status at least

#1 condition slightly distressed

#2 condition medium problem

#3 condition too bad
cp_zero_target_zero=len(data[(data.Cp==0)&(data.Target==0)])

cp_zero_target_one=len(data[(data.Cp==0)&(data.Target==1)])
sns.barplot(x=['cp_zero_target_zero','cp_zero_target_one'],y=[cp_zero_target_zero,cp_zero_target_one])

plt.show()

cp_one_target_zero=len(data[(data.Cp==1)&(data.Target==0)])

cp_one_target_one=len(data[(data.Cp==1)&(data.Target==1)])



sns.barplot(x=['cp_one_target_zero','cp_one_target_one'],y=[cp_one_target_zero,cp_one_target_one])

plt.show()
cp_two_target_zero=len(data[(data.Cp==2)&(data.Target==0)])

cp_two_target_one=len(data[(data.Cp==2)&(data.Target==1)])

sns.barplot(x=['cp_two_target_zero','cp_two_target_one'],y=[cp_two_target_zero,cp_two_target_one])

plt.show()
cp_three_target_zero=len(data[(data.Cp==3)&(data.Target==0)])

cp_three_target_one=len(data[(data.Cp==3)&(data.Target==1)])



sns.barplot(x=['cp_three_target_zero','cp_three_target_one'],y=[cp_three_target_zero,cp_three_target_one])

plt.show()

target_0_agerang_0=len(data[(data.Target==0)&(data.AgeRange==0)])

target_1_agerang_0=len(data[(data.Target==1)&(data.AgeRange==0)])



colors = ['blue','green']

explode = [0,0]

plt.figure(figsize = (5,5))

plt.pie([target_0_agerang_0,target_1_agerang_0], explode=explode, labels=['Target 0 Age Range 0','Target 1 Age Range 0'], colors=colors, autopct='%1.1f%%')

plt.title('Target vs Age Range Young Age ',color = 'blue',fontsize = 15)

plt.show()
target_0_agerang_1=len(data[(data.Target==0)&(data.AgeRange==1)])

target_1_agerang_1=len(data[(data.Target==1)&(data.AgeRange==1)])

colors = ['blue','green']

explode = [0,0]

plt.figure(figsize = (5,5))

plt.pie([target_0_agerang_1,target_1_agerang_1], explode=explode, labels=['Target 0 Age Range 1','Target 1 Age Range 1'], colors=colors, autopct='%1.1f%%')

plt.title('Target vs Age Range Middle Age',color = 'blue',fontsize = 15)

plt.show()

target_0_agerang_2=len(data[(data.Target==0)&(data.AgeRange==2)])

target_1_agerang_2=len(data[(data.Target==1)&(data.AgeRange==2)])

colors = ['blue','green']

explode = [0,0]

plt.figure(figsize = (5,5))

plt.pie([target_0_agerang_2,target_1_agerang_2], explode=explode, labels=['Target 0 Age Range 2','Target 1 Age Range 2'], colors=colors, autopct='%1.1f%%')

plt.title('Target vs Age Range Elderly Age ',color = 'blue',fontsize = 15)

plt.show()
col = ['age', 'sex', 'cp', 'fbs', 'restecg', 'trestbps',

       'exang', 'oldpeak', 'slope', 'ca', 'thal']

plt.style.use('ggplot')

for item in col:

    pd.crosstab(df[item], df.target).plot(kind='bar', figsize=(15, 7))

    plt.title("{} with target".format(str(item)))

    plt.legend(["non disease", "disease"])

    plt.ylabel("Frequency")

plt.show()
# f, axes = plt.subplots(4,4, figsize=(20, 15))

# sb.distplot( heart["age"], ax=axes[0,0])

# sb.distplot( heart["sex"], ax=axes[0,1])

# sb.distplot( heart["cp"], ax=axes[0,2])

# sb.distplot( heart["trestbps"], ax=axes[0,3])

# sb.distplot( heart["chol"], ax=axes[1,0])

# sb.distplot( heart["fbs"], ax=axes[1,1])

# sb.distplot( heart["restecg"], ax=axes[1,2])

# sb.distplot( heart["thalach"], ax=axes[1,3])

# sb.distplot( heart["exang"], ax=axes[2,0])

# sb.distplot( heart["oldpeak"], ax=axes[2,1])

# sb.distplot( heart["slope"], ax=axes[2,2])

# sb.distplot( heart["ca"], ax=axes[2,3])

# sb.distplot( heart["thal"], ax=axes[3,0])

# sb.distplot( heart["target"], ax=axes[3,1])

# plt.show()
plt.subplots(figsize=(20,10))

sns.heatmap(df.corr(), annot=True)

plt.show()
sns.barplot(x=data.Thalach.value_counts()[:20].index,y=data.Thalach.value_counts()[:20].values)

plt.xlabel('Thalach')

plt.ylabel('Count')

plt.title('Thalach Counts')

plt.xticks(rotation=45)

plt.show()
age_unique=sorted(data.Age.unique())

age_thalach_values=data.groupby('Age')['Thalach'].count().values

mean_thalach=[]

for i,age in enumerate(age_unique):

    mean_thalach.append(sum(data[data['Age']==age].Thalach)/age_thalach_values[i])
#data_sorted=data.sort_values(by='Age',ascending=True)

plt.figure(figsize=(10,5))

sns.pointplot(x=age_unique,y=mean_thalach,color='red',alpha=0.8)

plt.xlabel('Age',fontsize = 15,color='blue')

plt.xticks(rotation=45)

plt.ylabel('Thalach',fontsize = 15,color='blue')

plt.title('Age vs Thalach',fontsize = 15,color='blue')

plt.grid()

plt.show()
age_range_thalach=data.groupby('AgeRange')['Thalach'].mean()
sns.barplot(x=age_range_thalach.index,y=age_range_thalach.values)

plt.xlabel('Age Range Values')

plt.ylabel('Maximum Thalach By Age Range')

plt.title('illustration of the thalach to the age range')

plt.show()

#As shown in this graph, this rate decreases as the heart rate 

#is faster and in old age areas.

cp_thalach=data.groupby('Cp')['Thalach'].mean()
sns.barplot(x=cp_thalach.index,y=cp_thalach.values)

plt.xlabel('Degree of Chest Pain (Cp)')

plt.ylabel('Maximum Thalach By Cp Values')

plt.title('Illustration of thalach to degree of chest pain')

plt.show()

#As seen in this graph, it is seen that the heart rate is less 

#when the chest pain is low. But in cases where chest pain is 

#1, it is observed that the area is more. 2 and 3 were found to 

#be of the same degree.
sns.countplot(data.Thal)

plt.show()
data[(data.Thal==0)]
data[(data['Thal']==1)].Target.value_counts()

sns.barplot(x=data[(data['Thal']==1)].Target.value_counts().index,y=data[(data['Thal']==1)].Target.value_counts().values)

plt.xlabel('Thal Value')

plt.ylabel('Count')

plt.title('Counter for Thal')

plt.show()
#Target 1

a=len(data[(data['Target']==1)&(data['Thal']==0)])

b=len(data[(data['Target']==1)&(data['Thal']==1)])

c=len(data[(data['Target']==1)&(data['Thal']==2)])

d=len(data[(data['Target']==1)&(data['Thal']==3)])

print('Target 1 Thal 0: ',a)

print('Target 1 Thal 1: ',b)

print('Target 1 Thal 2: ',c)

print('Target 1 Thal 3: ',d)



#so,Apparently, there is a rate at Thal 2.Now, draw graph

print('*'*50)

#Target 0

e=len(data[(data['Target']==0)&(data['Thal']==0)])

f=len(data[(data['Target']==0)&(data['Thal']==1)])

g=len(data[(data['Target']==0)&(data['Thal']==2)])

h=len(data[(data['Target']==0)&(data['Thal']==3)])

print('Target 0 Thal 0: ',e)

print('Target 0 Thal 1: ',f)

print('Target 0 Thal 2: ',g)

print('Target 0 Thal 3: ',h)
f,ax=plt.subplots(figsize=(7,7))

sns.barplot(y=['T 1&0 Th 0','T 1&0 Th 1','T 1&0 Th 2','Ta 1&0 Th 3'],x=[1,6,130,28],color='green',alpha=0.5,label='Target 1 Thal State')

sns.barplot(y=['T 1&0 Th 0','T 1&0 Th 1','T 1&0 Th 2','Ta 1&0 Th 3'],x=[1,12,36,89],color='red',alpha=0.7,label='Target 0 Thal State')

ax.legend(loc='lower right',frameon=True)

ax.set(xlabel='Target State and Thal Counter',ylabel='Target State and Thal State',title='Target VS Thal')

plt.xticks(rotation=90)

plt.show()

#so, there has been a very nice graphic display. This is the situation that best describes the situation.
data.Target.unique()

#only two values are shown.

#A value of 1 is the value of patient 0.
sns.countplot(data.Target)

plt.xlabel('Target')

plt.ylabel('Count')

plt.title('Target Counter 1 & 0')

plt.show()

#determine the age ranges of patients with and without sickness and make analyzes about them

age_counter_target_1=[]

age_counter_target_0=[]

for age in data.Age.unique():

    age_counter_target_1.append(len(data[(data['Age']==age)&(data.Target==1)]))

    age_counter_target_0.append(len(data[(data['Age']==age)&(data.Target==0)]))



#now, draw show on graph    
#Target 1 & 0 show graph on scatter

plt.scatter(x=data.Age.unique(),y=age_counter_target_1,color='blue',label='Target 1')

plt.scatter(x=data.Age.unique(),y=age_counter_target_0,color='red',label='Target 0')

plt.legend(loc='upper right',frameon=True)

plt.xlabel('Age')

plt.ylabel('Count')

plt.title('Target 0 & Target 1 State')

plt.show()
male_young_t_1=data[(data['Sex']==1)&(data['AgeRange']==0)&(data['Target']==1)]

male_middle_t_1=data[(data['Sex']==1)&(data['AgeRange']==1)&(data['Target']==1)]

male_elderly_t_1=data[(data['Sex']==1)&(data['AgeRange']==2)&(data['Target']==1)]

print(len(male_young_t_1))

print(len(male_middle_t_1))

print(len(male_elderly_t_1))
f,ax1=plt.subplots(figsize=(20,10))

sns.pointplot(x=np.arange(len(male_young_t_1)),y=male_young_t_1.Trestbps,color='lime',alpha=0.8,label='Young')

sns.pointplot(x=np.arange(len(male_middle_t_1)),y=male_middle_t_1.Trestbps,color='black',alpha=0.8,label='Middle')

sns.pointplot(x=np.arange(len(male_elderly_t_1)),y=male_elderly_t_1.Trestbps,color='red',alpha=0.8,label='Elderly')

plt.xlabel('Range',fontsize = 15,color='blue')

plt.xticks(rotation=90)

plt.legend(loc='upper right',frameon=True)

plt.ylabel('Trestbps',fontsize = 15,color='blue')

plt.title('Age Range Values vs Trestbps',fontsize = 20,color='blue')

plt.grid()

plt.show()
for i,col in enumerate(data.columns.values):

    plt.subplot(5,3,i+1)

    plt.scatter([i for i in range(303)],data[col].values.tolist())

    plt.title(col)

    fig,ax=plt.gcf(),plt.gca()

    fig.set_size_inches(10,10)

    plt.tight_layout()

plt.show()

sns.heatmap(data.corr())
data.head()
data.corr()
import statsmodels.formula.api as sm

X = np.append(arr= np.ones((303,1)).astype(int), values=data, axis = 1)



X_l=data.iloc[:,[0,1,2,3,4,5,6]].values

r=sm.OLS(endog=data.iloc[:,-1:],exog=X_l).fit()

print(r.summary())

dataX=data.drop('Target',axis=1)

dataY=data['Target']
dataX.head()
dataY.head()
X_train,X_test,y_train,y_test=train_test_split(dataX,dataY,test_size=0.2,random_state=42)
print('X_train',X_train.shape)

print('X_test',X_test.shape)

print('y_train',y_train.shape)

print('y_test',y_test.shape)
#Normalization as the first process

# Normalize

X_train=(X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train)).values

X_test=(X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test)).values
pd.plotting.scatter_matrix(dataX,

                          c=['green','blue','red'],

                          figsize=[15,15],

                          diagonal='hist',

                          alpha=0.8,

                          s=200,

                          marker='*',

                          edgecolor='black')

plt.show()
from sklearn.decomposition import PCA

pca=PCA().fit(X_train)

print(pca.explained_variance_ratio_)

print()

print(X_train.columns.values.tolist())

print(pca.components_)
cumulative=np.cumsum(pca.explained_variance_ratio_)

plt.step([i for i in range(len(cumulative))],cumulative)

plt.show()
pca = PCA(n_components=8)

pca.fit(X_train)

reduced_data_train = pca.transform(X_train)

#inverse_data = pca.inverse_transform(reduced_data)

plt.scatter(reduced_data_train[:, 0], reduced_data_train[:, 1], label='reduced')

plt.xlabel('First Principal Component')

plt.ylabel('Second Principal Component')

plt.show()
pca = PCA(n_components=8)

pca.fit(X_test)

reduced_data_test = pca.transform(X_test)

#inverse_data = pca.inverse_transform(reduced_data)

plt.scatter(reduced_data_test[:, 0], reduced_data_test[:, 1], label='reduced')

plt.xlabel('First Principal Component')

plt.ylabel('Second Principal Component')

plt.show()
reduced_data_train = pd.DataFrame(reduced_data_train, columns=['Dim1', 'Dim2','Dim3','Dim4','Dim5','Dim6','Dim7','Dim8'])

reduced_data_test = pd.DataFrame(reduced_data_test, columns=['Dim1', 'Dim2','Dim3','Dim4','Dim5','Dim6','Dim7','Dim8'])

X_train=reduced_data_train

X_test=reduced_data_test

def plot_roc_(false_positive_rate,true_positive_rate,roc_auc):

    plt.figure(figsize=(5,5))

    plt.title('Receiver Operating Characteristic')

    plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],linestyle='--')

    plt.axis('tight')

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()

    

def plot_feature_importances(gbm):

    n_features = X_train.shape[1]

    plt.barh(range(n_features), gbm.feature_importances_, align='center')

    plt.yticks(np.arange(n_features), X_train.columns)

    plt.xlabel("Feature importance")

    plt.ylabel("Feature")

    plt.ylim(-1, n_features)
combine_features_list=[

    ('Dim1','Dim2','Dim3'),

    ('Dim4','Dim5','Dim5','Dim6'),

    ('Dim7','Dim8','Dim1'),

    ('Dim4','Dim8','Dim5')

]
parameters=[

{

    'penalty':['l1','l2'],

    'C':[0.1,0.4,0.5],

    'random_state':[0]

    },

]



for features in combine_features_list:

    print(features)

    print("*"*50)

    

    X_train_set=X_train.loc[:,features]

    X_test_set=X_test.loc[:,features]

    

    gslog=GridSearchCV(LogisticRegression(),parameters,scoring='accuracy')

    gslog.fit(X_train_set,y_train)

    print('Best parameters set:')

    print(gslog.best_params_)

    print()

    predictions=[

    (gslog.predict(X_train_set),y_train,'Train'),

    (gslog.predict(X_test_set),y_test,'Test'),

    ]

    

    for pred in predictions:

        print(pred[2] + ' Classification Report:')

        print("*"*50)

        print(classification_report(pred[1],pred[0]))

        print("*"*50)

        print(pred[2] + ' Confusion Matrix:')

        print(confusion_matrix(pred[1], pred[0]))

        print("*"*50)



    print("*"*50)    

    basari=cross_val_score(estimator=LogisticRegression(),X=X_train,y=y_train,cv=12)

    print(basari.mean())

    print(basari.std())

    print("*"*50) 
from sklearn.linear_model import LogisticRegression



lr=LogisticRegression(C=0.1,penalty='l1',random_state=0)

lr.fit(X_train,y_train)



y_pred=lr.predict(X_test)





y_proba=lr.predict_proba(X_test)



false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_proba[:,1])

roc_auc = auc(false_positive_rate, true_positive_rate)

plot_roc_(false_positive_rate,true_positive_rate,roc_auc)





from sklearn.metrics import r2_score,accuracy_score



#print('Hata Oranı :',r2_score(y_test,y_pred))

print('Accurancy Oranı :',accuracy_score(y_test, y_pred))

print("Logistic TRAIN score with ",format(lr.score(X_train, y_train)))

print("Logistic TEST score with ",format(lr.score(X_test, y_test)))

print()



cm=confusion_matrix(y_test,y_pred)

print(cm)

sns.heatmap(cm,annot=True)

plt.show()
parameters=[

{

    'n_neighbors':np.arange(2,33),

    'n_jobs':[2,6]

    },

]

print("*"*50)

for features in combine_features_list:

    print("*"*50)

    

    X_train_set=X_train.loc[:,features]

    X_test_set=X_test.loc[:,features]

   

    gsknn=GridSearchCV(KNeighborsClassifier(),parameters,scoring='accuracy')

    gsknn.fit(X_train_set,y_train)

    print('Best parameters set:')

    print(gsknn.best_params_)

    print("*"*50)

    predictions = [

    (gsknn.predict(X_train_set), y_train, 'Train'),

    (gsknn.predict(X_test_set), y_test, 'Test1')

    ]

    for pred in predictions:

        print(pred[2] + ' Classification Report:')

        print("*"*50)

        print(classification_report(pred[1], pred[0]))

        print("*"*50)

        print(pred[2] + ' Confusion Matrix:')

        print(confusion_matrix(pred[1], pred[0]))

        print("*"*50)

        

    print("*"*50)    

    basari=cross_val_score(estimator=KNeighborsClassifier(),X=X_train,y=y_train,cv=12)

    print(basari.mean())

    print(basari.std())

    print("*"*50)
knn=KNeighborsClassifier(n_jobs=2, n_neighbors=22)

knn.fit(X_train,y_train)



y_pred=knn.predict(X_test)



y_proba=knn.predict_proba(X_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_proba[:,1])

roc_auc = auc(false_positive_rate, true_positive_rate)

plot_roc_(false_positive_rate,true_positive_rate,roc_auc)



from sklearn.metrics import r2_score,accuracy_score



print('Accurancy Oranı :',accuracy_score(y_test, y_pred))

print("KNN TRAIN score with ",format(knn.score(X_train, y_train)))

print("KNN TEST score with ",format(knn.score(X_test, y_test)))

print()



cm=confusion_matrix(y_test,y_pred)

print(cm)

sns.heatmap(cm,annot=True)

plt.show()
n_neighbors = range(1, 17)

train_data_accuracy = []

test1_data_accuracy = []

for n_neigh in n_neighbors:

    knn = KNeighborsClassifier(n_neighbors=n_neigh,n_jobs=5)

    knn.fit(X_train, y_train)

    train_data_accuracy.append(knn.score(X_train, y_train))

    test1_data_accuracy.append(knn.score(X_test, y_test))

plt.plot(n_neighbors, train_data_accuracy, label="Train Data Set")

plt.plot(n_neighbors, test1_data_accuracy, label="Test1 Data Set")

plt.ylabel("Accuracy")

plt.xlabel("Neighbors")

plt.legend()

plt.show()
n_neighbors = range(1, 17)

k_scores=[]

for n_neigh in n_neighbors:

    knn = KNeighborsClassifier(n_neighbors=n_neigh,n_jobs=5)

    scores=cross_val_score(estimator=knn,X=X_train,y=y_train,cv=12)

    k_scores.append(scores.mean())

print(k_scores)
plt.plot(n_neighbors,k_scores)

plt.xlabel('Value of k for KNN')

plt.ylabel("Cross-Validated Accurancy")

plt.show()
parameters = [

{

    'learning_rate': [0.01, 0.02, 0.002],

    'random_state': [0],

    'n_estimators': np.arange(3, 20)

    },

]

for features in combine_features_list:

    print("*"*50)

    X_train_set=X_train.loc[:,features]

    X_test1_set=X_test.loc[:,features]

   

    gbc = GridSearchCV(GradientBoostingClassifier(), parameters, scoring='accuracy')

    gbc.fit(X_train_set, y_train)

    print('Best parameters set:')

    print(gbc.best_params_)

    print("*"*50)

    predictions = [

    (gbc.predict(X_train_set), y_train, 'Train'),

    (gbc.predict(X_test1_set), y_test, 'Test1')

    ]

    for pred in predictions:

        print(pred[2] + ' Classification Report:')

        print("*"*50)

        print(classification_report(pred[1], pred[0]))

        print("*"*50)

        print(pred[2] + ' Confusion Matrix:')

        print(confusion_matrix(pred[1], pred[0]))

        print("*"*50)

        

    print("*"*50)    

    basari=cross_val_score(estimator=GradientBoostingClassifier(),X=X_train,y=y_train,cv=4)

    print(basari.mean())

    print(basari.std())

    print("*"*50)
bc=GradientBoostingClassifier(learning_rate=0.02,n_estimators=18,random_state=0)

gbc.fit(X_train,y_train)



y_pred=gbc.predict(X_test)



y_proba=gbc.predict_proba(X_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_proba[:,1])

roc_auc = auc(false_positive_rate, true_positive_rate)

plot_roc_(false_positive_rate,true_positive_rate,roc_auc)



from sklearn.metrics import r2_score,accuracy_score



print('Accurancy Oranı :',accuracy_score(y_test, y_pred))

print("GradientBoostingClassifier TRAIN score with ",format(gbc.score(X_train, y_train)))

print("GradientBoostingClassifier TEST score with ",format(gbc.score(X_test, y_test)))

print()



cm=confusion_matrix(y_test,y_pred)

print(cm)

sns.heatmap(cm,annot=True)

plt.show()
y = df.target

X = df.drop(['target'], axis=1).values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
algo = {'Logistic Regression': LogisticRegression(solver='liblinear'), 

        'Decision Tree':DecisionTreeClassifier(), 

        'Random Forest':RandomForestClassifier(n_estimators=10, random_state=0), 

        'SVM':SVC(gamma=0.01, kernel='linear')

       }

predict_value = {}

for k, v in algo.items():

    model = v

    model.fit(X_train, y_train)

    predict_value[k] = model.score(X_test, y_test)*100

    print('Acurracy of ' + k + ' is {0:.2f}'.format(model.score(X_test, y_test)*100))
plt.figure(figsize=(12, 7))

sns.barplot(x=list(predict_value.keys()), y=list(predict_value.values()))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Accuracy")

plt.xlabel("Modals")

plt.show()