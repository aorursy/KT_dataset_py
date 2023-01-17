import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
train = pd.read_csv("../input/train.csv",index_col = 'PassengerId')

train.describe()
test = pd.read_csv("../input/test.csv",index_col = 'PassengerId')

test.describe()
train['Age'].fillna(train['Age'].median(), inplace=True)

test['Age'].fillna(test['Age'].median(), inplace=True)

merged_df = train.append(test)

merged_df
 
merged_df['Mr'] = 0

merged_df['Mrs'] = 0

merged_df['Miss'] = 0

merged_df['royalty'] = 0

merged_df['officer'] = 0
for index,row in merged_df.iterrows():

    name = row['Name']

    if 'Mr.' in name:

        merged_df.set_value(index,'Mr',1)

    elif 'Mrs.' in name:

        merged_df.set_value(index,'Mrs',1)

    elif 'Miss.' in name:

        merged_df.set_value(index,'Miss',1)

    elif 'Lady' or 'Don.' or 'Dona.' or 'Sir.' or 'Master.' in name:

        merged_df.set_value(index,'royalty',1)

    elif 'Rev' in name:

        merged_df.set_value(index,'officer',1)

        

merged_df
merged_df.drop('Name',inplace=True, axis=1)

merged_df.head() #Dropped the names column
merged_df['Embarked_S'] = 0

merged_df['Embarked_C'] = 0

merged_df['Embarked_Q'] = 0

merged_df['Embarked_unknown'] = 0



for index,row in merged_df.iterrows():

    embarkment = row['Embarked']

    if embarkment == 'S':

        merged_df.set_value(index,'Embarked_S',1)

    elif embarkment == 'C':

        merged_df.set_value(index,'Embarked_C',1)

    elif embarkment == 'Q':

        merged_df.set_value(index,'Embarked_Q',1)

    else:

        merged_df.set_value(index,'Embarked_unknown',1)

   



merged_df.head()
merged_df.drop('Embarked', inplace = True, axis = 1) #Dropped column 'Embarked'
for index,row in merged_df.iterrows():

    if row['Sex'] == 'male':

        merged_df.set_value(index, 'Sex', 1)

    else:

        merged_df.set_value(index,'Sex',0)

merged_df.head()
merged_df.drop('Ticket', inplace= True, axis = 1)

#lets categorize the fares as: cheap, average, and costly



merged_df['Fare_cheap']=0

merged_df['Fare_average']=0

merged_df['Fare_costly']=0



for index,row in merged_df.iterrows():

    if row['Fare'] <= 30.0 :

        merged_df.set_value(index, 'Fare_cheap', 1)

    elif row['Fare'] >30 and  row['Fare'] <= 70.0:

        merged_df.set_value(index,'Fare_average',1)

    else:

        merged_df.set_value(index, 'Fare_costly',1)

        

merged_df.head()
merged_df.drop('Fare',inplace = True, axis =1) #now we don't need the fare column

merged_df.head()
#I wont be considering the feature 'Cabin' 

#So,dropping that column as well

merged_df.drop('Cabin',inplace = True, axis = 1)

merged_df.head()
merged_df.describe() #Checking for any missing values due to manipulation
training_set = merged_df[:891]

training_set.describe()
testing_set = merged_df[891:]

testing_set.describe()
X = training_set[['Pclass','Sex','Age','SibSp','Parch','Mr','Mrs','Miss','royalty','officer','Embarked_S','Embarked_C','Embarked_Q','Embarked_unknown','Fare_cheap','Fare_average','Fare_costly']]

y = training_set.Survived #Works if there aren't any spaces in the column name



#17 features

X.shape
y.shape
from sklearn.svm import SVC

from sklearn.cross_validation import cross_val_score #k fold cross validation



svm_model = SVC() 

svm_model.kernel= 'linear'

score_svm = cross_val_score(svm_model,X,y,cv=10, scoring= 'accuracy')

print(score_svm.mean())
svm_model.fit(X,y)
X_new = testing_set[['Pclass','Sex','Age','SibSp','Parch','Mr','Mrs','Miss','royalty','officer','Embarked_S','Embarked_C','Embarked_Q','Embarked_unknown','Fare_cheap','Fare_average','Fare_costly']]

y_predict = svm_model.predict(X_new)
y_predict
prediction_df = pd.DataFrame({'PassengerId':testing_set.index,'Survived': y_predict})

prediction_df
submission_df = prediction_df.set_index('PassengerId')
submission_df.shape
submission_df['Survived']=submission_df['Survived'].astype(np.int) #change the datatype of column "Survived"

print(submission_df.dtypes)

submission_df.to_csv("submission_rs.csv")

submission_df