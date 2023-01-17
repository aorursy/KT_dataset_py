import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
ttnc_train = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')
# ttnc_train
ttnc_test = pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')
# ttnc_test
ttnc_gender = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
# ttnc_gender
train_df = pd.DataFrame(ttnc_train)
test_df = pd.DataFrame(ttnc_test)
train_df.head()
test_df.head()
desc = train_df.isnull()
desc.tail()
train_df.info()
test_df.info()
labelencoder= LabelEncoder()
sex = labelencoder.fit_transform(train_df['Sex'])
ticket = labelencoder.fit_transform(train_df['Ticket'])
sex1 = labelencoder.fit_transform(test_df['Sex'])
ticket1 = labelencoder.fit_transform(test_df['Ticket'])

test_df['Sex'] = sex1
test_df['Ticket'] = ticket1
train_df['Sex'] = sex
train_df['Ticket'] = ticket
age = train_df['Age']
age_miss = age.mean()
age_miss
Fare = test_df['Fare']
fare_chn = Fare.mean()
fare_chn
Fare.fillna(fare_chn, inplace = True)
Fare.describe()
age1 = test_df['Age']
age_miss1 = age1.mean()
age_miss1
age_miss = int(age_miss)
age.fillna(age_miss,inplace=True)
age.describe()
age_miss1 = int(age_miss1)
age1.fillna(age_miss1,inplace=True)
age1.describe()
train_df['Cabin'].unique()
train_df['Embarked'].unique()
test_df['Cabin'].unique()
test_df['Embarked'].unique()
train_df.drop(['Cabin','Embarked','Name'], axis=1, inplace=True)
test_df.drop(['Cabin','Embarked','Name'], axis=1, inplace=True)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(train_df['Age'], train_df['Fare'], train_df['Parch'], cmap=plt.cm.jet, linewidth=0.01)
ax.view_init(30,45)
plt.xlabel('Survived')
plt.ylabel('Sex')
plt.show()
sns.heatmap(train_df.corr())
sns.set_style('darkgrid')
sns.countplot('Survived', hue='Sex', data=train_df)
sns.pairplot(train_df)
x_train = train_df.drop('Survived',axis=1)
y_train = train_df['Survived'].values
logi_reg = LogisticRegression()
logi_reg.fit(x_train,y_train)
pred = logi_reg.predict(test_df)
pred
Prediction = pd.DataFrame(pred)
Prediction.head(10)

Prediction[0].unique()
Prediction.replace([0,1], ['No','Yes'], inplace=True)
import os
Prediction.to_csv('Prediction.csv')