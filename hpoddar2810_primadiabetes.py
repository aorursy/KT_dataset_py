
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')#, names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])
print(data.shape)
data.head()
data.tail()
data.columns
data.describe()
data.info()
columns=data.columns
columns
def distplot(column):
  df = data[column]
  sns.distplot(df)
  plt.show()

sns.set_style('darkgrid')
for column in columns:
  distplot(column)
df = data[(data.Glucose == 0) | (data.BloodPressure==0) | (data.Insulin == 0) | (data.SkinThickness==0) | (data.BMI ==0)]
df.describe()
def box(column):
  sns.boxplot(x='Outcome', y=column, data=data)
  plt.show()
for column in columns:
  box(column)
def violin(column):
  sns.violinplot(x='Outcome', y=column, data=data)
  plt.show()
for column in columns:
  violin(column)
def scatter(x, y):
  sns.scatterplot(x=x, y=y, hue='Outcome', data=data, marker='x')
  plt.show()
for i in range(1,8):
  scatter(columns[0], columns[i])
df = pd.DataFrame(index=data.columns)
for column in columns:
  df.loc[column, 'count'] = int(len(data[data[column] == 0]))
df
update_column = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in update_column:
  data.loc[(data[column] == 0) & (data['Outcome'] == 0), column] = data[data.Outcome == 0][column].mean()
  data.loc[(data[column] == 0) & (data['Outcome'] == 1), column] = data[data.Outcome == 1][column].mean()
data
data.describe()
for column in columns:
  violin(column)
from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(data[columns[:-1]], data.iloc[:,-1], random_state=77)
from sklearn.tree import DecisionTreeClassifier

dec_model = DecisionTreeClassifier()
dec_model.fit(x_train, y_train)
print("Train Accuracy: ", dec_model.score(x_train,y_train))
print("Test Accuracy: ", dec_model.score(x_test, y_test))
%%time
from sklearn.model_selection import GridSearchCV

params = {'criterion':['gini', 'entropy'],
          'max_depth': [5, 10, 20, 25, 30],
          'max_features': [3, 5, 7, 9],
          'max_leaf_nodes': [2,5,6, 9, 10, 15],
          'splitter': ['best', 'random']}
grid = GridSearchCV(DecisionTreeClassifier(), params, cv=10)
grid.fit(x_train, y_train)
print(grid.best_params_)
print("Score: ", grid.best_score_)
#grid
grid.score(x_test, y_test)
from sklearn import tree
plt.figure(figsize=(18, 10))
tree.plot_tree(grid.best_estimator_, filled=True)
grid
grid.best_estimator_
pd.Series(grid.best_estimator_.feature_importances_, index=columns[:8]).nlargest(8).plot(kind='barh')

x_train_updated = x_train[['Glucose', 'Age', 'Insulin']]
x_test_updated = x_test[['Glucose', 'Age', 'Insulin']]

params = {'criterion':['gini', 'entropy'],
          'max_depth': [5, 10, 20, 25, 30],
          'max_features': [3, 5, 7, 9],
          'max_leaf_nodes': [2,5,6, 9, 10, 15],
          'splitter': ['best', 'random']}
grid = GridSearchCV(DecisionTreeClassifier(), params, cv=10)
grid.fit(x_train_updated, y_train)
print(grid.best_params_)
print("Score: ", grid.best_score_)
grid.score(x_test_updated, y_test)
%%time
from sklearn.ensemble import RandomForestClassifier

params = params = {'criterion':['gini', 'entropy'],
          'max_depth': [5, 10, 20, 25, 30],
          'max_features': [3, 5, 7, 9],
          'max_leaf_nodes': [2,5,6, 9, 10, 15],
          #'splitter': ['best', 'random'],
          'n_estimators':[1,3,5,10]}
random_grid = GridSearchCV(RandomForestClassifier(), params, cv=10)
random_grid.fit(x_train, y_train)
print(random_grid.best_params_)
print(random_grid.best_score_)
random_grid.score(x_test, y_test)
clf = random_grid.best_estimator_
pd.Series(random_grid.best_estimator_.feature_importances_, columns[:8]).nlargest(8).plot(kind='barh')
from sklearn.metrics import confusion_matrix 
con_matrix = confusion_matrix(y_test, clf.predict(x_test))

sns.heatmap(con_matrix, annot=True, fmt='g')
plt.xticks([.5,1.5], ['No', 'Yes'])
plt.yticks([1.5,0.5],['Yes', 'No'],)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matric');
data['Outcome'].value_counts()
%%time
from sklearn.ensemble import RandomForestClassifier

params = params = {'criterion':['gini', 'entropy'],
          'max_depth': [5, 10, 20, 25, 30],
          'max_features': [3, 5, 7, 9],
          'max_leaf_nodes': [2,5,6, 9, 10, 15],
          #'splitter': ['best', 'random'],
          'n_estimators':[1,3,5,10],
          'class_weight':['balanced']}
random_grid = GridSearchCV(RandomForestClassifier(), params, cv=10)
random_grid.fit(x_train, y_train)
print(random_grid.best_params_)
print(random_grid.best_score_)
random_grid.score(x_test, y_test)
clf = random_grid.best_estimator_
cmatrix = confusion_matrix(y_test,clf.predict(x_test))
sns.heatmap(cmatrix, annot=True, fmt='g')
plt.xticks([.5,1.5], ['No', 'Yes'])
plt.yticks([1.5,0.5],['Yes', 'No'],)
plt.xlabel('Predicted')

plt.ylabel('Actual')
plt.title('Confusion Matric');
from sklearn.utils import resample, shuffle
data_pos = data[data.Outcome == 1]
data_neg = data[data.Outcome == 0]
data_pos = resample(data_pos, n_samples=500, random_state=34)
data1 = pd.concat([data_pos, data_neg], axis=0)
new_data = shuffle(data1, random_state=34)
new_data.describe()
x_train, x_test,y_train, y_test = train_test_split(new_data[columns[:-1]], new_data.iloc[:,-1], random_state=77)
%%time
from sklearn.ensemble import RandomForestClassifier

params = params = {'criterion':['gini', 'entropy'],
          'max_depth': [5, 10, 20, 25, 30],
          'max_features': [3, 5, 7, 9],
          'max_leaf_nodes': [2,5,6, 9, 10, 15],
          #'splitter': ['best', 'random'],
          'n_estimators':[1,3,5,10]}
random_grid = GridSearchCV(RandomForestClassifier(), params, cv=10)
random_grid.fit(x_train, y_train)
print(random_grid.best_params_)
print(random_grid.best_score_)
random_grid.score(x_test, y_test)
clf = random_grid.best_estimator_
cmatrix = confusion_matrix(y_test,clf.predict(x_test))
sns.heatmap(cmatrix, annot=True, fmt='g')
plt.xticks([.5,1.5], ['No', 'Yes'])
plt.yticks([1.5,0.5],['Yes', 'No'],)
plt.xlabel('Predicted')

plt.ylabel('Actual')
plt.title('Confusion Matric');
%%time
from sklearn.svm import SVC
params = {'C':[0.5,1,10,100],
          'gamma':['scale', 1, 0.1,0.01, 0.001, 0.0001],
          'kernel':['rbf']}

grid = GridSearchCV(SVC(), params, cv=10, scoring='accuracy')
grid.fit(x_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
grid.score(x_test,y_test)
cmatrix = confusion_matrix(y_test, grid.predict(x_test))
sns.heatmap(cmatrix, annot=True, fmt='g')
plt.xticks([.5,1.5], ['No', 'Yes'])
plt.yticks([1.5,0.5],['Yes', 'No'],)
plt.xlabel('Predicted')

plt.ylabel('Actual')
plt.title('Confusion Matric');
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix 
%%time
print("wwe")
from sklearn.svm import SVC
params = {'C':[0.5,1,10,100],
          'gamma':['scale', 1, 0.1,0.01, 0.001, 0.0001],
          'kernel':['rbf'],
          'class_weight':['balanced']}

grid = GridSearchCV(SVC(), params, cv=10, scoring='accuracy')
grid.fit(x_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
grid.score(x_test,y_test)
cmatrix = confusion_matrix(y_test, grid.predict(x_test))
sns.heatmap(cmatrix, annot=True, fmt='g')
plt.xticks([.5,1.5], ['No', 'Yes'])
plt.yticks([1.5,0.5],['Yes', 'No'],)
plt.xlabel('Predicted')

plt.ylabel('Actual')
plt.title('Confusion Matric');
print("wwe")
from sklearn.svm import SVC
params = {'C':[0.5,1,10,100],
          'gamma':['scale', 1, 0.1,0.01, 0.001, 0.0001],
          'kernel':['rbf', 'sigmoid']}

grid = GridSearchCV(SVC(), params, cv=10, scoring='accuracy')
grid.fit(x_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
grid.score(x_test, y_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, random_grid.predict(x_test), target_names=['No', 'Yes']))
print(classification_report(y_test, grid.predict(x_test), target_names=['No', 'Yes']))
