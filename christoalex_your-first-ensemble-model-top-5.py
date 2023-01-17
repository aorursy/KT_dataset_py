import pandas as pd



import numpy as np



from sklearn.ensemble import RandomForestClassifier, VotingClassifier



from xgboost import XGBClassifier



from sklearn.svm import SVC



import matplotlib.pyplot as plt



%matplotlib inline





titanic_filepath = ('../input/titanic/train.csv')



titanic_data= pd.read_csv(titanic_filepath)



test_filepath = ('../input/titanic/test.csv')



test_data=pd.read_csv(test_filepath)



titanic_data.head()
features=['Sex', 'Pclass','Parch','SibSp','Fare']

x=titanic_data[features]



test_x=test_data[features]
y=titanic_data.Survived
x.head()


cleanup_nums = {"Sex":     {"male": 1, "female": 2}}



cleanup_nums2 = {"Embarked":     {"S": 1, "C": 2, "Q": 3}}



x.head()
x.replace(cleanup_nums, inplace=True)



x.head()



test_x.replace(cleanup_nums, inplace=True)



x.replace(cleanup_nums2, inplace=True)



x.head()



test_x.replace(cleanup_nums2, inplace=True)



x.head()

x=x.fillna(x.mean())



test_x=test_x.fillna(test_x.mean())

rf=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)



svm=SVC()



model=XGBClassifier()

model= VotingClassifier(estimators=[ ('rf', rf), ('xgb', model), ('svm', svm)], voting='hard')



model.fit(x,y)


submission_path = ('../input/titanic/gender_submission.csv')



submission= pd.read_csv(submission_path)
submission['Survived']=model.predict(test_x)




submission['PassengerId']=test_data['PassengerId']



submission.columns=['PassengerId','Survived']





submission.columns=['PassengerId','Survived']



submission.head()

submission.to_csv('Submission.csv', index=False)