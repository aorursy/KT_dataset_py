import numpy as np

import pandas as pd



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
train_df = pd.read_csv("../input/titanic/train.csv", header=0)

test_df = pd.read_csv("../input/titanic/test.csv", header=0)

ids = test_df["PassengerId"].values





train_df = train_df[['Survived', 'Pclass', 'Sex', 'Fare']]

train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())









encoder_sex = LabelEncoder()

train_df['Sex'] = encoder_sex.fit_transform(train_df['Sex'].values)

standard = StandardScaler()

train_df_std = pd.DataFrame(standard.fit_transform(train_df[['Pclass', 'Fare']]), columns=['Pclass', 'Fare'])



train_df['Pclass'] = train_df_std['Pclass']

train_df['Fare'] = train_df_std['Fare']



train_x = train_df.drop('Survived',axis = 1)

train_y = train_df.Survived
#

#

test_df = test_df[['Pclass', 'Sex', 'Fare']]

test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

encoder_sex = LabelEncoder()

test_df['Sex'] = encoder_sex.fit_transform(test_df['Sex'].values)

standard = StandardScaler()

test_df_std = pd.DataFrame(standard.fit_transform(test_df[['Pclass', 'Fare']]), columns=['Pclass', 'Fare'])



test_df['Pclass'] = test_df_std['Pclass']

test_df['Fare'] = test_df_std['Fare']

test_x = test_df


model = SVC(random_state=1,max_iter=5000)

model.fit(train_x, train_y)



pred = model.predict(test_x)
file_submit = "titanic_logistic_regression.csv"

#

dft = pd.DataFrame({'PassengerId': ids, 'Survived': pred})

dft.to_csv(file_submit,index=False)

dft.head()