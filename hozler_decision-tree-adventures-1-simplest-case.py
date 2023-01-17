import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import graphviz 
import numpy as np
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dataset = pd.read_csv("../input/StudentsPerformance.csv")
dataset.head()
ax = sns.countplot(x="gender", data=dataset)
ax = sns.countplot(x="race/ethnicity", data=dataset)
ax = sns.countplot(y="parental level of education", data=dataset)
ax = sns.countplot(x="lunch", data=dataset)
ax = sns.countplot(x="test preparation course", data=dataset)
ax = sns.distplot(dataset["math score"]);
dataset["math grade"] = ""
dataset.loc[(dataset["math score"] >= 60), "math grade"] = "Pass"
dataset.loc[(dataset["math score"] < 60), "math grade"] = "Fail"
dataset.drop(columns=['math score', 'reading score', 'writing score'], inplace=True)
dataset.head()
one_hot = pd.get_dummies(dataset['gender'], prefix='gender', drop_first=True)
dataset = dataset.join(one_hot)
one_hot = pd.get_dummies(dataset['race/ethnicity'], prefix='race/ethnicity', drop_first=True)
dataset = dataset.join(one_hot)
one_hot = pd.get_dummies(dataset['parental level of education'], prefix='parental level of education', drop_first=True)
dataset = dataset.join(one_hot)
one_hot = pd.get_dummies(dataset['lunch'], prefix='lunch', drop_first=True)
dataset = dataset.join(one_hot)
one_hot = pd.get_dummies(dataset['test preparation course'], prefix='test preparation course', drop_first=True)
dataset = dataset.join(one_hot)
dataset.head()
data_train, data_test_hold = train_test_split(dataset, test_size=0.30, random_state=21)
data_test, data_hold = train_test_split(data_test_hold, test_size=0.33, random_state=21)
columns_move = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course", "gender_male", "race/ethnicity_group B", "race/ethnicity_group C", "race/ethnicity_group D", "race/ethnicity_group E", "parental level of education_bachelor's degree", "parental level of education_high school", "parental level of education_master's degree", "parental level of education_some college", "parental level of education_some high school", "lunch_standard", "test preparation course_none"]
y_train = data_train["math grade"].values
X_train = data_train[columns_move].values
y_test = data_test["math grade"].values
X_test = data_test[columns_move].values
model = DecisionTreeClassifier(criterion='gini', splitter='best', 
                               max_depth=None, min_samples_split=2, 
                               min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                               max_features=None, random_state=None, 
                               max_leaf_nodes=None, min_impurity_decrease=0.0, 
                               min_impurity_split=None, class_weight=None, 
                               presort=False)
model.fit(X_train[:,5:], y_train)
y_pred = model.predict(X_test[:,5:])
print("Model Accuracy: %.2f" % (accuracy_score(y_test,y_pred)*100), "%")
a = pd.DataFrame(confusion_matrix(y_test,y_pred), columns=['prediction/fail', 'prediction/pass'], index=['actual/fail', 'actual/pass'])
print("Confusion Matrix:")
print(a)
print("Classification Report:")
print("")
print(classification_report(y_test,y_pred))
y_pred_train = model.predict(X_train[:,5:])
print("Model Accuracy: %.2f" % (accuracy_score(y_train,y_pred_train)*100), "%")
dot_data = StringIO()
export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
columns_move.append("math grade test")
columns_move.append("math grade pred")
y_pred = y_pred.reshape(len(y_pred),1)
y_test = y_test.reshape(len(y_test),1)
resultarray = np.append(X_test, y_test, axis=1)
resultarray = np.append(resultarray, y_pred, axis=1)
resultdf = pd.DataFrame(resultarray, columns=columns_move)
resultdf.drop(columns=["gender_male", "race/ethnicity_group B", "race/ethnicity_group C", "race/ethnicity_group D", "race/ethnicity_group E", "parental level of education_bachelor's degree", "parental level of education_high school", "parental level of education_master's degree", "parental level of education_some college", "parental level of education_some high school", "lunch_standard", "test preparation course_none"], inplace=True)
resultdf.head(200)