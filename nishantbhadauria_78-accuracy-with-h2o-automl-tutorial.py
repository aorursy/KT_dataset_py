import pandas as pd

titanic_train=pd.read_csv('../input/titanic/train.csv')

titanic_test=pd.read_csv('../input/titanic/test.csv')

titanic_train.describe()
titanic_train.isnull().sum()
titanic_test.isnull().sum()
titanic_train['Age'].fillna(titanic_train['Age'].median(), inplace = True)

titanic_test['Age'].fillna(titanic_test['Age'].median(), inplace = True)

titanic_train['Embarked'].fillna(titanic_train['Embarked'].mode(),inplace=True)

titanic_test['Fare'].fillna(titanic_test['Fare'].median(), inplace = True)
titanic_train['familysize']=titanic_train['SibSp']+titanic_train['Parch']+1

titanic_train['Solo']=(titanic_train['familysize'] >1 ).astype(int)

titanic_train['Solo'],titanic_train['familysize']
titanic_test['familysize']=titanic_test['SibSp']+titanic_test['Parch']+1

titanic_test['Solo']=(titanic_test['familysize'] >1 ).astype(int)

titanic_train['Solo'].value_counts()
Age_wise_survival=pd.DataFrame(titanic_train['Survived'].groupby(titanic_train['Age']).sum())

Age_wise_dist=pd.DataFrame(titanic_train['Survived'].groupby(titanic_train['Age']).count())

Age_wise_dist
import matplotlib.pyplot as plt

plt.plot(Age_wise_survival,label="Survived")

plt.plot(Age_wise_dist,label="Total")

plt.legend()
titanic_train['AgeBin'] = pd.cut(titanic_train['Age'].astype(int), 5)

titanic_train['AgeBin'].values
titanic_test['AgeBin'] = pd.cut(titanic_test['Age'].astype(int), 5)
Gender_wise_survival=pd.DataFrame(titanic_train['Survived'].groupby(titanic_train['Sex']).sum())

Gender_wise_count=pd.DataFrame(titanic_train['Survived'].groupby(titanic_train['Sex']).count())

Gender_wise_survival/Gender_wise_count
titanic_train['vip'] = [1 if x =='Master' else 0 for x in titanic_train['Name']] 

titanic_test['vip'] = [1 if x =='Master' else 0 for x in titanic_test['Name']] 
Fare_wise_dist=pd.DataFrame(titanic_train['Survived'].groupby(titanic_train['Fare']).count())

Fare_wise_survival=pd.DataFrame(titanic_train['Survived'].groupby(titanic_train['Fare']).sum())

Fare_wise_s_per=(Fare_wise_survival/Fare_wise_dist)*100

import matplotlib.pyplot as plt

plt.plot(Fare_wise_s_per,'bo',label="% of survived by fare")

plt.legend()
titanic_train['farebin'] = pd.cut(titanic_train['Fare'].astype(int), 3)

titanic_train['farebin'].values
titanic_test['farebin'] = pd.cut(titanic_test['Fare'].astype(int), 3)
y="Survived"

train_x=titanic_train[['Pclass','Sex','familysize','Solo','AgeBin','vip','farebin','Survived']]

test_x=titanic_test[['Pclass','Sex','familysize','Solo','AgeBin','vip','farebin']]
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

train_x=train_x.apply(lambda col: label.fit_transform(col), axis=0, result_type='expand')

test_x=test_x.apply(lambda col: label.fit_transform(col), axis=0, result_type='expand')

train_x
import h2o

from h2o.automl import H2OAutoML

h2o.init()
trframe=h2o.H2OFrame(train_x)
teframe=h2o.H2OFrame(test_x)
titanic_model = H2OAutoML(max_runtime_secs = 120, seed = 1, project_name = "titanic_kaggle_nishant")

titanic_model.train(y = y, training_frame = trframe)
titanic_model.leaderboard
titanic_model.leader.params.keys()
titanic_model.leader.params['colsample_bytree'],titanic_model.leader.params['stopping_rounds']
lb=titanic_model.leader
m = h2o.get_model(lb)
pred_h2o = titanic_model.leader.predict(teframe)

pred_h2o
pred_pandas=pred_h2o.as_data_frame(use_pandas=True)

pred_pandas
pred_pandas['Survived'] = [1 if x > 0.5 else 0 for x in pred_pandas['predict']] 

pred_pandas
output= titanic_test.merge(pred_pandas['Survived'], left_index=True, right_index=True)

output
output_final=output[['PassengerId','Survived']]

output.to_csv('GBM_NISHANT.csv',index="FALSE")
