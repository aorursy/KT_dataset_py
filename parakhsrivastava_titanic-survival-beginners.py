import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
train_D=pd.read_csv("../input/train.csv")
test_D=pd.read_csv("../input/test.csv")
print(train_D.head())
print(test_D.head())
train_D.isnull().sum()
train_D = train_D.drop(['Cabin'], axis = 1)
#train_df.columns
train_D= train_D.drop(['PassengerId','Name','Ticket'], axis=1)
#train_df.columns
max_age=train_D['Age'].max()
#print(min_age)
min_age=train_D['Age'].min()
null_values_count=177
age_fill_values=np.random.randint(min_age,max_age,size=null_values_count)
#print(age_fill_values)
train_D['Age'][np.isnan(train_D['Age'])]= age_fill_values
train_D.isnull().sum()
train_D['Embarked'].unique()
print(train_D['Embarked'][train_D['Embarked']=='S'].count())
print(train_D['Embarked'][train_D['Embarked']=='C'].count())
print(train_D['Embarked'][train_D['Embarked']=='Q'].count())
train_D['Embarked']=train_D['Embarked'].fillna('S')
pclass_df= train_D[['Survived','Pclass']]
fig= plt.figure(figsize=(10,5))
axes0= plt.subplot(1,2,1)
axes0= sns.countplot(x='Pclass',hue='Survived',data=pclass_df)
axes0.set_title('Number of Survival,Class based')
axes0.set_ylabel('Count')
axes0.legend(['No','Yes'])
survival_pclass=pclass_df.groupby(['Pclass'],as_index=False).mean()
axes1= plt.subplot(1,2,2)
axes1= sns.barplot(x='Pclass',y='Survived',data=survival_pclass)
axes1.set_title('Survival Rate Based on Class')
axes1.set_ylabel('Survival Rate')
plt.tight_layout()
train_D['High Class']=np.where(train_D['Pclass']==1,1,0)
train_D['Median Class']=np.where(train_D['Pclass']==2,1,0)
train_D.head()
sex_df=train_D[['Survived','Sex']]
fig=plt.figure(figsize=(10,5))
axes0= plt.subplot(1,2,1)
axes0= sns.countplot(x='Sex',hue='Survived',data=sex_df)
axes0.set_title('Number of Survival,Sex based')
axes0.set_ylabel('Count')
axes0.legend(['No','Yes'])
survival_sex=sex_df.groupby(['Sex'],as_index=False).mean()
axes1= plt.subplot(1,2,2)
axes1= sns.barplot(x='Sex',y='Survived',data=survival_sex,palette='Set1')
axes1.set_title('Survival Rate Based on Sex')
axes1.set_ylabel('Survival Rate')
#plt.tight_layout()
train_D = train_D.replace({'Sex':{'male':0,'female':1}})
train_D.head()
age_df = train_D[['Survived', 'Age']]
age_df['Age'] = pd.cut(age_df['Age'], bins = 5, labels = [1, 2, 3, 4, 5])
fig = plt.figure(figsize = (10,5))
axes0 = plt.subplot(1,2,1)
axes0 = sns.countplot(x = 'Age', hue = 'Survived', data = age_df)
axes0.set_title('The Number of Survival Based on Age')
axes0.set_ylabel('Count')
axes0.legend(['No', 'Yes'])

survival_age = age_df.groupby(['Age'], as_index = False).mean()
axes1 = plt.subplot(1,2,2)
axes1 = sns.barplot(x = 'Age', y = 'Survived', data = survival_age, palette = 'Set1') 
axes1.set_title('The Survival Rate Based on Age')
axes1.set_ylabel('Survival Rate')
age = train_D[['Age']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
age_scaled = min_max_scaler.fit_transform(age)
train_D['Age'] = pd.DataFrame(age_scaled)
train_D.head()
sibsp_df = train_D[['Survived', 'SibSp']]
survival_sibsp = sibsp_df.groupby(['SibSp'], as_index = False).mean()
sns.barplot(x = 'SibSp', y = 'Survived', data = survival_sibsp)
sibsp_df['SibSp'] = np.where(sibsp_df['SibSp'] > 0, 1, 0)
fig = plt.figure(figsize = (10,5))
axes0 = plt.subplot(1,2,1)
axes0 = sns.countplot(x = 'SibSp', hue = 'Survived', data = sibsp_df)
axes0.set_title('The Number of Survival Based on SibSp')
axes0.set_ylabel('Count')
axes0.legend(['No', 'Yes'])

survival_sibsp = sibsp_df.groupby(['SibSp'], as_index = False).mean()
axes1 = plt.subplot(1,2,2)
axes1 = sns.barplot(x = 'SibSp', y = 'Survived', data = survival_sibsp, palette = 'Set1') 
axes1.set_title('The Survival Rate Based on SibSp')
axes1.set_ylabel('Survival Rate')
train_D['SibSp'] = np.where(train_D['SibSp'] > 0, 1, 0)
train_D.head()
fare_df = train_D[['Survived', 'Fare']]
fare_df['Fare'] = pd.cut(fare_df['Fare'], bins = 5, labels = [1, 2, 3, 4, 5])
fig = plt.figure(figsize = (10,5))
axes0 = plt.subplot(1,2,1)
axes0 = sns.countplot(x = 'Fare', hue = 'Survived', data = fare_df)
axes0.set_title('The Number of Survival Based on Fare')
axes0.set_ylabel('Count')
axes0.legend(['No', 'Yes'])

survival_fare = fare_df.groupby(['Fare'], as_index = False).mean()
axes1 = plt.subplot(1,2,2)
axes1 = sns.barplot(x = 'Fare', y = 'Survived', data = survival_fare, palette = 'Set1') 
axes1.set_title('The Survival Rate Based on Fare')
axes1.set_ylabel('Survival Rate')
fare = train_D[['Fare']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
fare_scaled = min_max_scaler.fit_transform(fare)
train_D['Fare'] = pd.DataFrame(fare_scaled)
train_D.head()
embarked_df = train_D[['Survived', 'Embarked']]
fig = plt.figure(figsize = (10,5))
axes0 = plt.subplot(1,2,1)
axes0 = sns.countplot(x = 'Embarked', hue = 'Survived', data = embarked_df)
axes0.set_title('The Number of Survival Based on Embark')
axes0.set_ylabel('Count')
axes0.legend(['No', 'Yes'])

survival_embarked = embarked_df.groupby(['Embarked'], as_index = False).mean()
axes1 = plt.subplot(1,2,2)
axes1 = sns.barplot(x = 'Embarked', y = 'Survived', data = survival_embarked, palette = 'Set1') 
axes1.set_title('The Survival Rate Based on Embark')
axes1.set_ylabel('Survival Rate')
train_D['Embarked C'] = np.where(train_D['Embarked'] == 'C', 1, 0)
train_D['Embarked Q'] = np.where(train_D['Embarked'] == 'Q', 1, 0)
train_D.head()
final = train_D[['Survived', 'High Class', 'Median Class', 'Sex', 'Age', 'SibSp', 
                           'Parch', 'Fare', 'Embarked C', 'Embarked Q']]
final.head()
independent_v_train = final[['High Class', 'Median Class', 'Sex', 'Age', 'SibSp', 
                                      'Parch', 'Fare', 'Embarked C', 'Embarked Q']]
dependent_v_train = final['Survived']
independent_v_train.isnull().sum()
dt = DecisionTreeClassifier()
dt.fit(independent_v_train, dependent_v_train)
dt.score(independent_v_train, dependent_v_train)
# Transform Pclass and Sex variable
test_D['High Class'] = np.where(test_D['Pclass'] == 1, 1, 0)
test_D['Median Class'] = np.where(test_D['Pclass'] == 2, 1, 0)
test_D = test_D.replace({'Sex': {'male':0, 'female':1}})
# Fill NAs in Age and normalizing values
min_age = test_D['Age'].min()
max_age = test_D['Age'].max()
null_values_count = test_D['Age'].isnull().sum()
age_fill_values = np.random.randint(min_age, max_age, size = null_values_count)
test_D['Age'][np.isnan(test_D['Age'])] = age_fill_values
age = test_D[['Age']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
age_scaled = min_max_scaler.fit_transform(age)
test_D['Age'] = pd.DataFrame(age_scaled)
# Fill NA in Fare and Normalizing values
test_D['Fare'] = test_D['Fare'].fillna(test_D['Fare'].mean())
fare = train_D[['Fare']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
fare_scaled = min_max_scaler.fit_transform(fare)
test_D['Fare'] = pd.DataFrame(fare_scaled)
# Fill NA in Embarked and transform values
print(test_D['Embarked'][test_D['Embarked'] == 'S'].count()) # 270
print(test_D['Embarked'][test_D['Embarked'] == 'C'].count()) # 102
print(test_D['Embarked'][test_D['Embarked'] == 'Q'].count()) # 46
test_D['Embarked'] = test_D['Embarked'].fillna('S')
test_D['Embarked C'] = np.where(test_D['Embarked'] == 'C', 1, 0)
test_D['Embarked Q'] = np.where(test_D['Embarked'] == 'Q', 1, 0)
test_D.count()
final_test = test_D[['High Class', 'Median Class', 'Sex', 'Age', 'SibSp', 
                           'Parch', 'Fare', 'Embarked C', 'Embarked Q']]
final_test.head()
independent_v_test = final_test
dependent_v_test_predict = dt.predict(independent_v_test)
survival_df = pd.DataFrame(dependent_v_test_predict)
test_get_id = pd.read_csv('../input/test.csv')
prediction_df = pd.DataFrame(test_get_id['PassengerId'])
prediction_df['Survived'] = survival_df
prediction_df.to_csv('Prediction of Titanic.csv', index=False)