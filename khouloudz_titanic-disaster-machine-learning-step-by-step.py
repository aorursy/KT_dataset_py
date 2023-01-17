# Loading Numpy and Pandas Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns



import pandas as pd
## Creating panda dataframes from train and test CSV files
print("Loading Training and Testing Data =====>")
training_data = pd.read_csv('../input/train.csv')
testing_data = pd.read_csv('../input/test.csv')
gender_submission_data = pd.read_csv('../input/gender_submission.csv')
print("<===== Training and Testing Data Loading finished")
gender_submission_data.describe()
""" the testing data lacks of the column Survived which is present in the gender_submission file as the output
of the learning process later for the test part. Since we need to verify how good is our data before moving to
the learning process, i decided to build a new testing data that has the same shape as the training data
"""
testing_data = pd.merge(testing_data, gender_submission_data, on='PassengerId')
testing_data.head(5)
"""we need to reorder the data so that we have the same columns as in the training data"""
testing_data = testing_data[training_data.columns.tolist()]
testing_data.head(5)
'''
    Printing the 5 first samples in training_data dataframe 
'''
training_data.head(5)
'''
    Printing the 6 samples select randomly in training_data dataframe 
'''
training_data.sample(6)
training_data.describe()
training_data.columns
training_data.dtypes

%matplotlib inline
'''
    Creating dataframes separating survived and not survived passergers
'''
td_not_survived=training_data.loc[(training_data['Survived']==0)]
td_survived=training_data.loc[(training_data['Survived']==1)]
td_not_survived.head(5)
td_survived.sample(10)
f,ax = plt.subplots(3,4,figsize=(20,16))
sns.countplot('Pclass',data=training_data,ax=ax[0,0])
sns.countplot('Sex',data=training_data,ax=ax[0,1])
sns.boxplot(x='Pclass',y='Age',data=training_data,ax=ax[0,2])
sns.countplot('SibSp',hue='Survived',data=training_data,ax=ax[0,3],palette='husl')
sns.distplot(training_data['Fare'].dropna(),ax=ax[2,0],kde=False,color='b')
sns.countplot('Embarked',data=training_data,ax=ax[2,2])

sns.countplot('Pclass',hue='Survived',data=training_data,ax=ax[1,0],palette='husl')
sns.countplot('Sex',hue='Survived',data=training_data,ax=ax[1,1],palette='husl')
sns.distplot(training_data[training_data['Survived']==0]['Age'].dropna(),ax=ax[1,2],kde=False,color='r',bins=5)
sns.distplot(training_data[training_data['Survived']==1]['Age'].dropna(),ax=ax[1,2],kde=False,color='g',bins=5)
sns.countplot('Parch',hue='Survived',data=training_data,ax=ax[1,3],palette='husl')
sns.swarmplot(x='Pclass',y='Fare',hue='Survived',data=training_data,palette='husl',ax=ax[2,1])
sns.countplot('Embarked',hue='Survived',data=training_data,ax=ax[2,3],palette='husl')

ax[0,0].set_title('Total Passengers by Class')
ax[0,1].set_title('Total Passengers by Gender')
ax[0,2].set_title('Age Box Plot By Class')
ax[0,3].set_title('Survival Rate by SibSp')
ax[1,0].set_title('Survival Rate by Class')
ax[1,1].set_title('Survival Rate by Gender')
ax[1,2].set_title('Survival Rate by Age')
ax[1,3].set_title('Survival Rate by Parch')
ax[2,0].set_title('Fare Distribution')
ax[2,1].set_title('Survival Rate by Fare and Pclass')
ax[2,2].set_title('Total Passengers by Embarked')
ax[2,3].set_title('Survival Rate by Embarked')

df = training_data.groupby(['Sex','Survived']).size() # output pandas.core.series.Series
type(df) # pandas.core.series.Series
#df=df.unstack()
df.head()

plt.figure();df.plot(kind='bar').set_title('Gender histogram training data')
df = td_survived.groupby('Sex').size()
#df=df.unstack()
df.head()
plt.figure();df.plot(kind='bar').set_title('Survived passengers by gender');
df = td_not_survived.groupby('Sex').size()
plt.figure();df.plot(kind='bar').set_title(' Not Survived passengers by gender');
df = td_survived.groupby('Pclass').size()
plt.figure();df.plot(kind='bar').set_title('Survived passengers by Pclass');
df = td_not_survived.groupby('Pclass').size()
plt.figure();df.plot(kind='bar').set_title('Not Survived passengers by Pclass');
plt.figure();
td_survived.Age.hist()
'''
Let us see how the age is distributed over ALL the passengers whether survived or not
'''
plt.figure();
plt.suptitle("Passengers Age distribution",x=0.5, y=1.05, ha='center', fontsize='xx-large');
pl1 = training_data.Age.hist();
pl1.set_xlabel("Age")
pl1.set_ylabel("Count")


'''
 separate the passengers into 4 groups depending on Age:
'''
df_children = training_data.loc[(training_data['Age']>=0)].loc[(training_data['Age']<=15)]
df_y_adults = training_data.loc[(training_data['Age'] >15)].loc[(training_data['Age']<=30 )]
df_adults = training_data.loc[(training_data['Age'] >30)].loc[(training_data['Age']<=60 )]
df_old = training_data.loc[(training_data['Age'] >60)]
plt.figure(1)

df1 = df_children.groupby('Survived').size() # with .size() we generate a pandas pandas.core.series.Series Series type variable
plt.subplot(2,2,1)
df1.plot(kind='bar').set_title('Children') 
df2 = df_y_adults.groupby('Survived').size() # with .size() we generate a pandas  pandas.core.series.Series Series type variable
plt.subplot(2,2,2)
df2.plot(kind='bar').set_title('young Adults')
df3 = df_adults.groupby('Survived').size() # with .size() we generate a pandas pandas.core.series.Series Series type variable
plt.subplot(2,2,3)
df3.plot(kind='bar').set_title('Adults')
df4 = df_old.groupby('Survived').size() # with .size() we generate a pandas pandas.core.series.Series Series type variable
plt.subplot(2,2,4)
df4.plot(kind='bar').set_title('old')
f,ax = plt.subplots(2,2,figsize=(10,10))
sns.countplot('Survived',data=df_children,ax=ax[0,0])
sns.countplot('Survived',data=df_y_adults,ax=ax[0,1])
sns.countplot('Survived',data=df_adults,ax=ax[1,0])
sns.countplot('Survived',data=df_old,ax=ax[1,1])

ax[0,0].set_title('Survival Rate by children')
ax[0,1].set_title('Survival Rate by young adults')
ax[1,0].set_title('Survival Rate by adults')
ax[1,1].set_title('Survival Rate by old')

df_full = pd.concat([training_data,testing_data]) # axis : {0/’index’, 1/’columns’}, default 0 The axis to concatenate along (by index)
num_all = len(df_full.index)
''' number of records of training data'''
num_train = len(training_data.index)
''' number of records of testing data'''
num_test = len(testing_data.index)
d = {'full' : num_all, 'train' : num_train, 'test' : num_test}
number_records = pd.Series(d)
number_records.head()
df_sum_null = df_full.isnull().sum().sort_values(ascending=False) # output pandas.core.series.Series
#df=df_sum_null.unstack() ==> does not work
plt.figure();df_sum_null.plot(kind='barh') # showing a horizontal bar plot 
def percentage_missing(colname,df):
    length_df = len(df.index)
    per_missing = df_full[colname].isnull().sum() * 100/length_df
    return per_missing
type(df_full.columns)
id_col = df_full.columns.tolist()
type(d)
dict_missing = {}
for colname in id_col :
    d = {colname:percentage_missing(colname,df_full)}
    dict_missing.update(d)
missing_values_percent = pd.Series(dict_missing).sort_values(ascending=False) 
plt.figure(figsize=(10,5));
plt1 = missing_values_percent.plot(kind='bar')
plt.suptitle("% missing by column",x=0.5, y=1.05, ha='center', fontsize='xx-large');
plt1.set_ylabel("% missing")
print(dict_missing)
def data_clean_completeness(data):
    data.Age.fillna(value=data.Age.mean(), inplace=True)
    data.Fare.fillna(value=data.Fare.mean(), inplace=True)
    data.Cabin.fillna(value=(data.Cabin.value_counts().idxmax()), inplace=True)
    data.Embarked.fillna(value=(data.Embarked.value_counts().idxmax()), inplace=True)
    return data
training_data = data_clean_completeness(training_data)
testing_data = data_clean_completeness(testing_data)