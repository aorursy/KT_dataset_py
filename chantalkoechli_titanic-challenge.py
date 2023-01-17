import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl #visualization
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')



df_train.head()

df_test.head()
#How do variables differ by survival (across sex)? 

summary1 = df_train.groupby(['Survived']).mean()

summary1

#Pclass and Fare seem to differ 
#How do variables differ by survival when sex is controlled for? 

summary2 = df_train.groupby(['Survived', 'Sex']).mean()

summary2

#Fare seems to differ across survival state, but not between sex within survival class

#Ticket class seems to differ more for females across survival state, but does also differ for males

#Age differs in opposite directions for males and females across survival states

#Females that did not survive had many more siblings/spouses and parents/children on board

#Overall, gender matters - and interactions between gender and ticket class/age/sibsp/Parch are seen
#Confirming gender matters - survival rate much higher for females

summary = df_train.groupby(['Sex']).mean()

summary
#Are any variables correlated? Perhaps SibSp and Parch, or Pclass and Fare?

df_train.plot.scatter(x='Pclass', y='Fare')
#Are any variables correlated? Perhaps SibSp and Parch, or Pclass and Fare?

df_train.plot.scatter(x='SibSp', y='Parch')
#Looking for NAs in data

df_train.isna().sum()
#removing cabin variable (variable does not matter for model)

df_train2 = df_train.drop(columns = ['Cabin', 'Name', 'Ticket', 'Embarked'])

df_train2.head()
from sklearn.impute import SimpleImputer



df_train2 = pd.get_dummies(df_train2) #encoding categorical variables with dummy variables

my_imputer = SimpleImputer()

df_train3 = pd.DataFrame(my_imputer.fit_transform(df_train2))

df_train3.columns = df_train2.columns

df_train3.head()
from sklearn.model_selection import train_test_split



#Establishing x and y variables 

factors1 = ['Sex_female', 'Sex_male', 'Age', 'Fare', 'SibSp', 'Parch']

factors2 = ['Age', 'Fare', 'SibSp', 'Parch']

X1 = df_train3[factors1]



X2 = df_train3[factors2]

y1 = df_train3['Survived']



#splitting data

train_X1, val_X1, train_y1, val_y1 = train_test_split(X1, y1, random_state = 0)

train_X2, val_X2, train_y2, val_y2 = train_test_split(X2, y1, random_state = 0)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error





model1 = RandomForestClassifier(random_state=1, n_estimators=100)

model1.fit(train_X1, train_y1)

predictions1 = model1.predict(val_X1)



model2 = RandomForestClassifier(random_state=1)

model2.fit(train_X2, train_y2)

predictions2 = model2.predict(val_X2)



print(mean_absolute_error(val_y1, predictions1))

print(mean_absolute_error(val_y2, predictions2))



#model 1 is a better model
df_test2 = df_test.drop(columns = ['Cabin', 'Name', 'Ticket', 'Embarked'])

df_test3 = pd.get_dummies(df_test2) #encoding categorical variables with dummy variables

my_imputer = SimpleImputer()

df_test4 = pd.DataFrame(my_imputer.fit_transform(df_test3))

df_test4.columns = df_test3.columns





test_factors= df_test4[factors1] #subsetting test data to X variables

test_factors = pd.get_dummies(test_factors) #replacing categorical variables with dummy variables 



final_model = RandomForestClassifier(random_state=1, n_estimators =100) #setting up final model

final_model.fit(X1, y1)

final_predictions = final_model.predict(test_factors).astype(int)



output = pd.DataFrame({'PassengerId': df_test.PassengerId,

                      'Survived': final_predictions})

output.to_csv('submission.csv', index=False)

output.head()