import numpy as np

import pandas as pd 

import os

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

print(os.listdir("../input"))
df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



df['Sex'] = df.Sex.apply(lambda x : 1 if x=='male' else 0)

df_test['Sex'] = df_test.Sex.apply(lambda x : 1 if x=='male' else 0)



df['Embarked'] = df.Embarked.apply(lambda x: str(x))

df_test['Embarked'] = df_test.Embarked.apply(lambda x: str(x))



df.Age.fillna(29.699118, inplace = True)

df_test.Age.fillna(29.699118, inplace = True)

df_test.Fare.fillna(35, inplace = True)

df.head()
df.Embarked.unique()
# ohe = OneHotEncoder()

# X = ohe.fit_transform(df.Embarked.values.reshape(-1, 1))

# X_test = ohe.transform(df_test.Embarked.values.reshape(-1, 1))

# ohe.categories_
le = LabelEncoder()

X = le.fit_transform(df.Embarked.values.reshape(-1, 1))

X_test = le.transform(df_test.Embarked.values.reshape(-1, 1))

le.classes_
# ohe2 = OneHotEncoder()

# X_2 = ohe2.fit_transform(df.Sex.values.reshape(-1, 1))

# X_test_2 = ohe2.transform(df_test.Sex.values.reshape(-1, 1))

# ohe2.categories_
# df = df.drop('Embarked', axis = 1)

# df_test = df_test.drop('Embarked', axis = 1)



# df = df.drop('Sex', axis = 1)

# df_test = df_test.drop('Sex', axis = 1)
# df[['C', 'Q', 'S', 'nan']] = pd.DataFrame(X.toarray())

# df_test[['C', 'Q', 'S', 'nan']] = pd.DataFrame(X_test.toarray())

# df_test.head()
df['Embarked'] = pd.DataFrame(X)

df_test['Embarked'] = pd.DataFrame(X_test)

df.head()
# df[['female', 'male']] = pd.DataFrame(X_2.toarray())

# df_test[['female', 'male']] = pd.DataFrame(X_test_2.toarray())
df.head()
submi_df = df_test['PassengerId'].to_frame()

submi_df.head()
# features = ['Pclass', 'female', 'male', 'Age', 'SibSp', 'Parch', 'Fare', 'C', 'Q', 'S', 'nan' ]

# features = ['female', 'male', 'Age', 'SibSp', 'Parch', 'Fare', 'C', 'Q', 'S', 'nan' ] # Improved

# features = ['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'C', 'Q', 'S', 'nan' ] # Not Improved

# features = ['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked' ] # Not Improved

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked' ] # Not Improved





X = df[features]

X_test = df_test[features]

y = df.Survived
rf = RandomForestClassifier()

rf.fit(X, y)
y_pred = rf.predict(X_test)

y_pred[:5]
submi_df['Survived'] = y_pred

submi_df.to_csv('mysecondsubmissiontokagglecomp.csv', index_label = False, index = False)