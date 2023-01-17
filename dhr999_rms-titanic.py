#Import required libraries
from pandas import *
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from matplotlib.pyplot import *
%matplotlib inline
from numpy import *
from seaborn import *
from sklearn.model_selection import train_test_split
#Data Munging
data_train = read_csv('../input/train.csv')
data_test = read_csv('../input/test.csv')
test_data = data_test
x = data_train.drop(['Survived'],axis=1)
y = data_train.Survived
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33, random_state=0)
i=1
le = LabelEncoder()
for j in [x_train, x_test, data_test]:
    j = j.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
    j.Age = j.Age.fillna(j.Age.mean())
    j.Fare = j.Fare.fillna(j.Fare.mean())
    j.Embarked = j.Embarked.fillna('S')
    j.Embarked = le.fit_transform(j.Embarked)
    j.Sex = le.fit_transform(j.Sex)
    j['FamilySize'] = j.SibSp + j.Parch + 1
    j = j.drop(['SibSp','Parch'],axis=1)
    if i == 1:
        x_train = j
        i=2
    elif i == 2:
        x_test = j
        i=3
    elif i ==3:
        data_test = j
#Create Model
rfc = RandomForestClassifier(bootstrap = True,max_leaf_nodes = 82,random_state = 5,n_estimators = 1000)
rfc.fit(x_train,y_train)
#rfe = RFE(rfc, 4)
#rfe = rfe.fit(x_train, y_train)
# summarize the selection of the attributes
#print(rfe.support_)
#print(rfe.ranking_)
pred_val1 = rfc.predict(x_test)
pred_val = rfc.predict(data_test)
accuracy_score(pred_val1,y_test)
#Kaggle Submission
submission = DataFrame({"PassengerId":test_data['PassengerId'],'Survived':pred_val})

submission.to_csv('submission1.csv',index = False)