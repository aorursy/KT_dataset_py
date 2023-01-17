import pandas as pd

train_data_csv = pd.read_csv("../input/train.csv")

test_data_csv = pd.read_csv("../input/test.csv")
def embarked_coding(x):

    if x == 'S':

        return 0

    elif x == 'C':

        return 1

    elif x == 'Q':

        return 2

    else:

        return 'NaN'

    

train_data_csv['Embarked_coded'] = [embarked_coding(x) for x in train_data_csv['Embarked']]



def sex_coding(x):

    if x =='male':

        return 0

    elif x =='female':

        return 1

    else:

        return 'NaN'

train_data_csv['Sex_coded'] = [sex_coding(x) for x in train_data_csv['Sex']]
target = train_data_csv.Survived

train_colomn = ['Pclass','Sex_coded','Age','SibSp','Parch','Fare','Embarked_coded']

train_data = train_data_csv[train_colomn]
from sklearn.preprocessing import Imputer
my_imputer = Imputer(strategy = 'median',axis = 0)

train_data = my_imputer.fit_transform(train_data)
from sklearn.ensemble import RandomForestClassifier
my_model =  RandomForestClassifier(max_depth = 5)

my_model.fit(train_data,target)
test_data_csv['Embarked_coded'] = [embarked_coding(x) for x in test_data_csv['Embarked']]

test_data_csv['Sex_coded'] = [sex_coding(x) for x in test_data_csv['Sex']]
test_data = test_data_csv[train_colomn]

test_data = my_imputer.fit_transform(test_data)
predicted_value = my_model.predict(test_data)
submission_data = pd.DataFrame({'PassengerId':test_data_csv.PassengerId,'Survived':predicted_value})
submission_data.to_csv('submission.csv',index=False)