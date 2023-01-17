import pandas as pd

data = pd.read_csv('../input/titanic/train.csv')

data_test = pd.read_csv('../input/titanic/test.csv')

data
data.describe()
import matplotlib.pyplot as plt
# survived vs deceased

fig = plt.figure(figsize=(18,6))

data.Survived.value_counts().plot(kind='bar')

plt.title('Survived vs deceased')
# male survived vs male deceased

data.Survived[data.Sex == 'male'].value_counts().plot(kind = 'bar')
# female survived vs female deceased

data.Survived[data.Sex == 'female'].value_counts().plot(kind = 'bar')
# gender vs survival

data[data.Survived == 1].Sex.value_counts().plot(kind = 'bar')
# survival within class

for x in [1,2,3]:

    data.Survived[data.Pclass == x].plot(kind="kde")
# low class male survived vs male deceased

data.Survived[(data.Sex == "male") & (data.Pclass == 3)].value_counts().plot(kind='bar')
# high class male survived vs male deceased

data.Survived[(data.Sex == "male") & (data.Pclass == 1)].value_counts().plot(kind='bar')
# low class female survived vs female deceased

data.Survived[(data.Sex == "female") & (data.Pclass == 3)].value_counts().plot(kind='bar')
# high class female survived vs female deceased

data.Survived[(data.Sex == "female") & (data.Pclass == 1)].value_counts().plot(kind='bar')
data.isnull().sum()
687/891, 177/891
data = data.drop(['Cabin'], axis=1)

data_test = data_test.drop(['Cabin'], axis=1)

data.head()
data.describe()
data['Embarked'].value_counts()
data['Age'] = data['Age'].fillna('29')

data_test['Age'] = data_test['Age'].fillna('29')

data.isnull().sum()

data['Embarked'] = data['Embarked'].fillna('S')

data_test['Embarked'] = data_test['Embarked'].fillna('S')

data.head()
data = data.drop(['Name', 'Ticket'],axis=1)

data_test = data_test.drop(['Name', 'Ticket'],axis=1)

data_test
data_test.isnull().sum()
data_test.Fare.value_counts()
data_test['Fare']=data_test['Fare'].fillna('8')
data_test.isnull().sum()
data
rdata = data.copy()

tdata = data_test.copy()



tdata['Sex'] = tdata['Sex'].map({'male' : 0, 'female' : 1})

tdata['Embarked'] = tdata['Embarked'].map({'S' : 0, 'C' : 1, 'Q' : 2})



rdata['Sex'] = rdata['Sex'].map({'male' : 0, 'female' : 1})

rdata['Embarked'] = rdata['Embarked'].map({'S' : 0, 'C' : 1, 'Q' : 2})



tdata
from sklearn.linear_model import LogisticRegression



train_input_data = rdata[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

target = rdata['Survived']



lr = LogisticRegression()



lr.fit(train_input_data, target)
test_data = tdata[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
predictions = lr.predict(test_data)

print(predictions[:10])
predictionsPD = pd.Series(predictions)

submission = pd.concat([tdata['PassengerId'], predictionsPD], axis=1)

submission=submission.rename(columns={0: "Survived"})

submission.to_csv('submission_file.csv', index = False)

submission
from sklearn import tree, model_selection

decision_tree = tree.DecisionTreeClassifier(random_state=1)

decision_tree = decision_tree.fit(train_input_data, target)

predictions = decision_tree.predict(test_data)
decision_tree_two = tree.DecisionTreeClassifier(

    max_depth = 7,

    min_samples_split = 2,

    random_state = 1)

decision_tree_two = decision_tree_two.fit(train_input_data, target)



scores = model_selection.cross_val_score(decision_tree_two, train_input_data, target, scoring='accuracy', cv=10)



prediction_two = decision_tree_two.predict(test_data)
tree.export_graphviz(decision_tree_two, feature_names=["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"], out_file="tree.dot")



import graphviz

from IPython.display import display



with open("tree.dot") as f:

    dot_graph = f.read()



display(graphviz.Source(dot_graph))
import pydot #pip install pydot



(graph,) = pydot.graph_from_dot_file('tree.dot')

graph.write_png('tree.png')