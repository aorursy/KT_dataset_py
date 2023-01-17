import pandas as pd

from pathlib import Path







titanic_train_data = pd.read_csv("../input/titanic/train.csv")

titanic_test_data =pd.read_csv("../input/titanic/test.csv")





titanic_train_data.columns
#titanic_train_data.groupby(['Sex']).Age.mean()

mean_age=[]

mean_age=titanic_train_data.groupby(['Sex']).Age.mean()



#data cleansing 

titanic_train_data.loc[titanic_train_data['Age'].isnull() & (titanic_train_data['Sex'] == "female"),['Age']]=mean_age['female']

titanic_train_data.loc[titanic_train_data['Age'].isnull() & (titanic_train_data['Sex'] == "male"),['Age']]=mean_age['male']

titanic_train_data=titanic_train_data.dropna(subset=['Embarked'])

titanic_train_data['Dependants']=titanic_train_data.apply(lambda row:row.SibSp+row.Parch,axis=1)

features=['Sex', 'Age', 'Pclass','Embarked','Fare','Dependants']

X_train=titanic_train_data[features]

y_train=titanic_train_data.Survived



from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import make_column_transformer



col_trans=make_column_transformer(

    (OneHotEncoder(), ['Sex', 'Embarked']),

    remainder='passthrough') 





#OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

#OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train['Sex']))

#OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid['Sex']))



col_trans.fit_transform(X_train)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

randfor=RandomForestClassifier()

#logreg = LogisticRegression(solver='lbfgs')

pipe = make_pipeline(col_trans, randfor)
cross_val_score(pipe, X_train, y_train, cv=2, scoring='accuracy').mean()
pipe.fit(X_train, y_train)
#titanic_file_path_Test = Path("D:/Kaggle/test.csv")

#titanic_test_data=pd.read_csv(titanic_file_path_Test)

#data cleansing 

mean_age_test=[]

mean_age_test=titanic_test_data.groupby(['Sex']).Age.mean()

titanic_test_data.loc[titanic_test_data['Age'].isnull() & (titanic_test_data['Sex'] == "female"),['Age']]=mean_age['female']

titanic_test_data.loc[titanic_test_data['Age'].isnull() & (titanic_test_data['Sex'] == "male"),['Age']]=mean_age['male']

titanic_test_data.loc[titanic_test_data['Fare'].isnull(),['Fare']]=titanic_test_data['Fare'].mean()

titanic_test_data['Dependants']=titanic_test_data.apply(lambda row:row.SibSp+row.Parch,axis=1)

#titanic_test_data=titanic_test_data.dropna(subset=['Embarked'])

features=['Sex', 'Age', 'Pclass','Embarked','Fare','Dependants']

X_test=titanic_test_data[features]



pipe.fit(X_test)

test_preds=pipe.predict(X_test)

output = pd.DataFrame({'PassengerId': titanic_test_data.PassengerId,

                       'Survived': test_preds})

output.to_csv('submission1.csv', index=False)