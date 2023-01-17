import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
train_data.head()
test_data.head()
print("******** TRAIN ********")
print(train_data.info())
print("******** TEST ********")
print(test_data.info())
train_data.drop(axis=1,columns=(["PassengerId","Cabin","Name"]),inplace=True)

submission_passengers = test_data.PassengerId

test_data.drop(axis=1,columns=(["PassengerId","Cabin","Name"]),inplace=True)
plt.figure(figsize=(10,5))
plt.title("Embarked CountPlot")
sns.countplot(data=train_data, x="Embarked")
plt.show()
train_data.Embarked.fillna(value="S",inplace=True)
train_data.fillna(value=-10,inplace=True)
test_data.fillna(value=-10,inplace=True)
print("******** TRAIN ********")
print(train_data.info())
print("******** TEST ********")
print(test_data.info())
train_y = train_data.Survived

train_data.drop(axis=1,columns=(["Survived"]),inplace=True)

train_x = train_data
test_x = test_data
cat_model = CatBoostClassifier(cat_features=[0,1,3,4,5,7])
cat_model.fit(train_x,train_y)
predictions = cat_model.predict(test_x)
my_submission = pd.DataFrame({'PassengerId': submission_passengers ,'Survived': (predictions.astype("Int64")) })
my_submission.to_csv("my_first_sub.csv",index=False)
