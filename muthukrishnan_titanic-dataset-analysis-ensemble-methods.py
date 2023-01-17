import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train_data = pd.read_csv("../input/train.csv")
train_data.head()
#selecting columns which are going to make sense to my analysis
train_data = train_data.loc[:,['Survived','Pclass','Sex']]

train_data = train_data.dropna()
#label encode the categorical data
data_for_analysis = pd.get_dummies(train_data, columns=["Sex","Pclass"])

features = data_for_analysis.iloc[:,1:].values
target = data_for_analysis.iloc[:,0].values

data_for_analysis.describe()
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10)
clf.fit(features, target)
#prediction
testcsv = pd.read_csv("../input/test.csv")

#selecting columns which are going to make sense to my analysis
test_data = testcsv.loc[:,['Pclass','Sex']]
test_features = pd.get_dummies(test_data, columns=["Sex","Pclass"])
predicted_survival = clf.predict(test_features.values)
from sklearn.metrics import classification_report
gender_submission = pd.read_csv("../input/gender_submission.csv")
print(classification_report(gender_submission.loc[:,'Survived'].values, predicted_survival))
submission = pd.DataFrame({'PassengerId':testcsv.loc[:,'PassengerId'].values, 'Survived': predicted_survival}).round().astype(int)
submission.to_csv('titanic_csv_submission.csv', index=False)