import pandas as pd

test_data = pd.read_csv ('/kaggle/input/titanic/test.csv')

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data['Survived'] = test_data.apply(lambda x: 0 if x.Sex=='male'else 1, axis=1)
test_data[['PassengerId','Survived']].to_csv('KaggleOutput', index=False)
from sklearn.tree import DecisionTreeClassifier



model=DecisionTreeClassifier().fit(train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']],train_data['Survived'])



model.predict(test_data['Survived'])

##And a big error is thrown.
for df in [train_data,test_data]:

    df['Sex_boolean']=df['Sex'].map({'male':1,'female':0})

    

model=DecisionTreeClassifier().fit(train_data[['Pclass', 'Sex_boolean', 'Age', 'SibSp', 'Parch', 'Fare']],train_data['Survived'])



model.predict(test_data['Survived'])
for df in [train_data, test_data]:

    df['Fare'].fillna(train_data['Fare'].mean(), inplace=True)

    df['Age'].fillna(train_data['Age'].mean(), inplace=True)



predictions=DecisionTreeClassifier().fit(train_data[['Pclass', 'Sex_boolean', 'Age', 'SibSp', 'Parch', 'Fare']],train_data['Survived']).predict(test_data[['Pclass', 'Sex_boolean', 'Age', 'SibSp', 'Parch', 'Fare']])



##Append the predictions to the PassengerID and convert to CSV

pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':predictions}).to_csv('KaggleOutput', index=False)
predictions=DecisionTreeClassifier().fit(train_data[['Pclass', 'Sex_boolean']],train_data['Survived']).predict(test_data[['Pclass', 'Sex_boolean']])



##Append the predictions to the PassengerID and convert to CSV

pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':predictions}).to_csv('KaggleOutput', index=False)
from sklearn import tree

import matplotlib.pyplot as plt



model=DecisionTreeClassifier().fit(train_data[['Pclass','Sex_boolean']],train_data['Survived'])



plt.figure(figsize=(40,20))  

_ = tree.plot_tree(model, feature_names = ['Pclass', 'Sex_boolean'], filled=True, fontsize=30, rounded = True)
##Scikit allows you to print the rules to see what is happening

from sklearn.tree.export import export_text



tree_rules = export_text(model, feature_names=['Pclass', 'Sex_boolean'])

print(tree_rules)