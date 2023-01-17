import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb

data = pd.read_csv('../input/train.csv', sep=',', na_values='.')#creating dataframe for training data

data1 = pd.read_csv('../input/test.csv', sep=',', na_values='.')#creating dataframe for test data

%matplotlib inline
data.head()
data1.head()
data.describe()
data1.describe()
#Dropping the columns that don't seem to be useful for the prediction

data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True, axis=1)

data1.drop(['Name', 'Ticket', 'Cabin'], inplace=True, axis=1)
temp = data['Embarked'].value_counts(ascending='True') #Finding the Frequency distribution for Embarked

print(temp)
temp1 = data['Sex'].value_counts(ascending='True') #Finding the Frequency distribution for Sex

print(temp1)
#Filling the missing values in Embarked 

data.Embarked.fillna('S', inplace=True)

data1.Embarked.fillna('S', inplace=True)
data["Age"].fillna(data["Age"].mean(), inplace=True)

data1["Age"].fillna(data1["Age"].mean(), inplace=True)

data.describe()
data1.describe()
#Cross_tabulation

Embarked_Survived=pd.crosstab(index=data["Embarked"],columns=data["Survived"],margins=True)

Embarked_Survived
observed=Embarked_Survived.ix[0:-1,0:-1]
#Cross_tabulation

Sex_Survived=pd.crosstab(index=data["Sex"],columns=data["Survived"],margins=True)

Sex_Survived
observed1=Sex_Survived.ix[0:-1,0:-1]
#Chi Square test

import scipy.stats as stats

stats.chi2_contingency(observed=observed)
import scipy.stats as stats

stats.chi2_contingency(observed=observed1)
face_age = sb.FacetGrid(data, hue='Survived', size=3,aspect=3)

face_age.map(sb.kdeplot, 'Age', shade=True)

face_age.set(xlim=(0, data.Age.max()))

face_age.add_legend()



#plt.subplots(1,1, figsize=(10,4))

#average_age = train[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()

#sb.barplot('Age', 'Survived', data=average_age)

#plt.xticks(rotation=90)

print ('Age survival relation: ')
plt.subplots(1,1, figsize=(15,5))

average_age = data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()

sb.barplot('Parch', 'Survived', data=average_age)

plt.xticks(rotation=90)

print ('Parch survival relation: ')
data.corr()

correlation_matrix=data.corr()

sb.heatmap(correlation_matrix, vmax=1., square=False).xaxis.tick_top()
#label encoding

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
categorical_data = pd.DataFrame()

categorical_data['Embarked']=data['Embarked']

categorical_data['Sex'] = data['Sex']



categorical_data.head(5)
encoded_cat_df = pd.DataFrame()

encoded_cat_df1 = pd.DataFrame()

for column in categorical_data.columns:

    le.fit(categorical_data[column])

    encoded_cat_df[column] = le.transform(categorical_data[column]) 

encoded_cat_df.head()
num_feature_list = [ 'Survived','Pclass','Age','SibSp','Parch','Fare']

num_df = (data[num_feature_list])

mydata = pd.concat([encoded_cat_df, num_df], axis=1)
mydata.describe()
cat_data = pd.DataFrame()

cat_data['Embarked']=data1['Embarked']

cat_data['Sex'] = data1['Sex']



cat_data.head(5)
encoded_cat_df2 = pd.DataFrame()

encoded_cat_df3 = pd.DataFrame()

for column in cat_data.columns:

    le.fit(cat_data[column])

    encoded_cat_df2[column] = le.transform(cat_data[column]) 

encoded_cat_df2.head()
num_feature_list1 = [ 'Pclass','Age','SibSp','Parch','Fare']

num_df1 = (data1[num_feature_list1])

testdata = pd.concat([encoded_cat_df2, num_df1], axis=1)
from sklearn.cross_validation import train_test_split

import numpy as np

from sklearn import linear_model

from sklearn import metrics

from sklearn.preprocessing import scale

X_train= mydata[['Sex','Age','Embarked','Parch','Pclass','SibSp']]

y_train = mydata['Survived']

X_test= testdata[['Sex','Age','Embarked','Parch','Pclass','SibSp']]

logr = linear_model.LogisticRegression()

X_train = np.array(X_train)

y_train = np.array(y_train)

# Train the model using the training sets

logr.fit(X_train, y_train)

y_pred = logr.predict(X_test)
output = pd.DataFrame({

        

        'PassengerId': data1.PassengerId,

        'Survived': y_pred

    })

output.to_csv('titanic_result.csv', index=False)