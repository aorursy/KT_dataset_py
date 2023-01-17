import numpy as np

import pandas as pd



from matplotlib import pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set_style('darkgrid')



import missingno as msno

import pickle
df = pd.read_csv('../input/titanic/train.csv')
df.head()
df.shape
df.dtypes
df.info()
df.describe()
df.describe(include=['object'])
sns.countplot(x = df.Survived, data = df)

plt.show()
df['Survived'].value_counts()
sns.countplot(x =df.Sex, data= df)

plt.show()
df.Sex.value_counts()
sns.countplot(x = df.Sex, hue = df['Survived'], data = df)

plt.show()
survived_df = df[df.Survived == 1]



male_survived = (survived_df[survived_df.Sex == 'male'].Survived.value_counts()  / survived_df.Survived.value_counts() ) * 100

female_survived = (survived_df[survived_df.Sex == 'female'].Survived.value_counts() / survived_df.Survived.value_counts()) * 100



print(f"Out of the Total Survived Population, Male Survived Population % = {male_survived}")

print(f"Out of the Total Survived Population, Female Survived Population % = {female_survived}")
sns.countplot(x = df.Pclass, data = df)
sns.countplot(x = df.Pclass, hue=df.Survived, data = df)

plt.show()
sns.countplot(x = df.Embarked, data = df)

plt.show()
sns.countplot(x = df.Embarked, hue = df.Survived, data = df)

plt.show()
grouped_data = df.groupby(['Embarked', 'Survived']).agg({'Survived': 'count'})

grouped_data['%'] = grouped_data.apply(lambda x: x*100/float(x.sum()))

grouped_data
df.Age.hist()
sns.boxplot(x = df.Age, data = df)
sns.boxplot(x = df.Sex, y = df.Age, hue= "Survived", data = df)

plt.show()
sns.boxplot(x = df.SibSp, data = df)

plt.show()
df.SibSp.value_counts()
sns.countplot(x = df.SibSp, hue = df.Survived, data = df)
df.Parch.value_counts()
sns.countplot(x = df.Parch, data = df)
sns.countplot(x = df.Parch, hue = df.Survived, data = df)

plt.show()
msno.matrix(df)
msno.bar(df)
df.isnull().mean() * 100
df.duplicated().sum()
correlations = df.corr()

correlations
sns.heatmap(data = correlations, cmap = 'RdBu_r')

plt.show()
df.head()
sns.boxplot(x = df.Age)

plt.show()
sns.violinplot(x = df.SibSp)

plt.show()
df.groupby('SibSp').agg(['count'])
sns.violinplot(x = df.Parch)
df.groupby('Parch').agg(['count'])
sns.boxplot(x = df.Fare)

plt.show()
sns.violinplot(x = df.Pclass , y = df.Age, hue = 'Survived', data = df)

plt.show()
df.isnull().mean() * 100
df.isnull().sum()/ len(df) 
df.isnull().sum()
df.head()
df[df.Embarked.isnull()]
first_class = df['Pclass'] == 1

female_people = df['Sex'] == 'female'

no_siblings = df['SibSp'] == 0

no_parents = df['Parch'] == 0
filtered_df = df[first_class & female_people & no_siblings & no_parents]

filtered_df.head()
filtered_df.Embarked.value_counts()
df_copy = df.drop('PassengerId', axis=1)
df_copy.fillna({

    'Embarked': 'C'

}, inplace=True)
df_copy.isnull().sum()
def plot_correlationmap(df):

  # Set the Plot Size

  plt.figure(figsize=(10, 10))

  sns.set_style('white')

  # Fetch the Correlation Matrix

  correlations = df_copy.corr()

  # Display 1 Half of the Map to Avoid Duplicate of HeatMap and Screen Clutter

  boolean_mask = np.zeros_like(correlations)

  upperTriangle = np.triu_indices_from(boolean_mask)

  boolean_mask[upperTriangle] = 1



  sns.heatmap(correlations * 100, cmap='RdBu_r', annot= True, fmt='0.0f', mask=boolean_mask)
plot_correlationmap(df_copy)
df_copy.Age = df.groupby('Pclass')['Age'].apply(lambda x: x.fillna(x.mean()))
df_copy.isnull().sum()
df_copy.describe(include=['object'])
df_copy.fillna(method='ffill', inplace=True)

df_copy.fillna(method='bfill', inplace=True)
df_copy.isnull().sum()
df_copy.describe(include=["object"])
df_copy.head()
numeric_features = df_copy.dtypes[df_copy.dtypes != object].index

numeric_features = np.delete(numeric_features, [0, 1])
for feature in numeric_features:

  sns.boxplot(x = feature, data = df)

  plt.show()
df_copy.head()
df_copy = df_copy.drop('Cabin', axis= 1)
df_copy = df_copy.drop('Name', axis=1)
df_copy.Ticket.describe()
df_copy = df_copy.drop('Ticket', axis=1)
df_copy.head()
from sklearn.model_selection import train_test_split
features = df_copy.drop('Survived', axis=1)

target = df_copy['Survived']
features_Train, features_Test, target_Train, target_Test = train_test_split(features, target, test_size= 0.30, random_state=987)
print('Features Train - ', features_Train.shape)

print('Features Test - ', features_Test.shape)

print('Target Train - ', target_Train.shape)

print('Target Test - ', target_Test.shape)
target_Train.value_counts(normalize=True) * 100
target_Test.value_counts(normalize=True) * 100
df_copy.dtypes
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelEncoder = LabelEncoder()

df_copy['Sex'] = labelEncoder.fit_transform(df_copy['Sex'])

df_copy.dtypes
df_copy.Sex.unique()
OHEncoder = OneHotEncoder()

embarked_columns_encoded = OHEncoder.fit_transform(df_copy['Embarked'].values.reshape(-1, 1)).toarray()

embarked_coded = pd.DataFrame(embarked_columns_encoded, columns=["Embarked_" + str(int(i)) for i in range(embarked_columns_encoded.shape[1])])

df_copy = df_copy.join(embarked_coded)

df_copy.head()
df_copy = df_copy.drop('Embarked', axis=1)
df_copy.head()
df.to_csv('final_titatic_data.csv', index=False)
df_copy = pd.read_csv('final_titatic_data.csv')
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, plot_confusion_matrix, roc_curve, confusion_matrix, plot_roc_curve

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.naive_bayes import GaussianNB
df = pd.read_csv('final_titatic_data.csv')

df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

features = df.drop('Survived', axis=1)

target = df.Survived

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.30, random_state=1234)
df.head()
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

print(Y_train.value_counts(normalize=True), Y_test.value_counts(normalize=True))
pipelines = {

    'tree': make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=1234)),

    'rf': make_pipeline(StandardScaler(), RandomForestClassifier(random_state=1234)),

    'lr': make_pipeline(StandardScaler(), LogisticRegression(random_state=1234)),

    'svm': make_pipeline(StandardScaler(), svm.SVC(random_state=1234)),

    'gb': make_pipeline(StandardScaler(), GaussianNB())

}


tree_hyperparameters = {

    'decisiontreeclassifier__splitter': ['best', 'random'],

    'decisiontreeclassifier__max_features': list(range(1, X_train.shape[1])),

    'decisiontreeclassifier__min_samples_split': np.linspace(1, 10, 10, endpoint=True),

    'decisiontreeclassifier__min_samples_leaf': np.linspace(0.1, 0.5, 10, endpoint=True),

    'decisiontreeclassifier__max_depth': np.linspace(1, 32, 32, endpoint=True)

}

rf_hyperparameters = {

    'randomforestclassifier__min_samples_split': np.linspace(1, 10, 10, endpoint=True),

    'randomforestclassifier__min_samples_leaf': np.linspace(0.1, 0.5, 10, endpoint=True),

    'randomforestclassifier__max_depth': np.linspace(10, 110, 11),

    'randomforestclassifier__max_features': ['auto', 'sqrt'],

    'randomforestclassifier__bootstrap': [True, False]

}

svm_hyperparameters = {

    'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001],

    'svc__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],

    'svc__C': [0.1, 1, 10, 100, 1000]

}



lr_hyperparameters = {

    'logisticregression__penalty': ['l1', 'l2', 'elasticnet', 'none'],

    'logisticregression__C': np.linspace(-4, 4, 20),

    'logisticregression__max_iter': [100, 1000, 10000],

    'logisticregression__solver': ['lbfgs','newton-cg','liblinear','sag','saga']

}



gb_hyperparameters = {

    

}



hyperparameters = {

    'tree': tree_hyperparameters,

    'rf': rf_hyperparameters,

    'svm': svm_hyperparameters,

    'lr': lr_hyperparameters,

    'gb': gb_hyperparameters

}
fitted_models = {}



for name, pipeline in pipelines.items():

  model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)

  model.fit(X_train, Y_train)

  print(name, "model is fitted")

  fitted_models[name] = model
# Fit the LR Model



model = GridSearchCV(pipelines['lr'], hyperparameters['lr'], cv=10, n_jobs=-1)

model.fit(X_train, Y_train)

print("Logistic Regression Model is fitted")
predicted = model.predict(X_test)

print(accuracy_score(Y_test, predicted))
svmModel = GridSearchCV(pipelines['svm'], hyperparameters['svm'], cv=10, n_jobs=-1)

svmModel.fit(X_train, Y_train)

print("SVM Model is fitted")
prdicted = svmModel.predict(X_test)

print(accuracy_score(Y_test, prdicted))
import pickle



with open('fitted_models.pkl', 'wb') as f:

  pickle.dump(fitted_models, f)
for name, model in fitted_models.items():

  print('Model - ', name)

  pred = model.predict(X_test)

  print("Accuracy Score -", accuracy_score(Y_test, pred))

  print()
model_performance = {}



for name, model in fitted_models.items():

  model_performance[name] = {}

  pred = model.predict(X_test)

  model_performance[name]['Accuracy Score'] = accuracy_score(Y_test, pred)

  model_performance[name]['Precision Score'] = precision_score(Y_test, pred)

  model_performance[name]['Recall Score'] = precision_score(Y_test, pred)

  model_performance[name]['F1 Score'] = f1_score(Y_test, pred)



 
performance_list = [['Model', 'Accuracy Score', 'Precision Score', 'Recall Score', 'F1 Score']]

for name, performance in model_performance.items():

  metrics = list()

  metrics.append(name)

  metrics.append(performance['Accuracy Score'])

  metrics.append(performance['Precision Score'])

  metrics.append(performance['Recall Score'])

  metrics.append(performance['F1 Score'])

  performance_list.append(metrics)



column_names = performance_list.pop(0)



df_performance = pd.DataFrame(performance_list, columns=column_names)

df_performance.Model.replace('tree', 'Decision Tree', inplace=True)

df_performance.Model.replace('rf', 'Random Forest', inplace=True)

df_performance.Model.replace('lr', 'Logistic Regression', inplace=True)

df_performance.Model.replace('svm', 'Support Vector Machines', inplace=True)

df_performance.Model.replace('gb', 'Gaussian Naive Bayes', inplace=True)

df_performance.head()
def model_train(model, model_name):

  model.fit(X_train, Y_train)

  predicted = model.predict(X_test)

  confusion = confusion_matrix(Y_test, predicted)

  plot_confusion_matrix(model, X_test, Y_test, display_labels=['Survived', 'Not Survived'], cmap=plt.cm.Blues)

  print(f"{model_name} - ")

  print("Accuracy - ", accuracy_score(Y_test, predicted))

  print("Sensitivity/ Recall Score - ", recall_score(Y_test, predicted))

  print("Precision Score - ", precision_score(Y_test, predicted))

  print("F1 Score - ", f1_score(Y_test, predicted))

  print("ROC Curve - ")

  plot_roc_curve(model, X_test, Y_test)

  plt.show()

  return model
target_Test.value_counts(normalize=True) * 100
unrestricted_model = DecisionTreeClassifier(random_state=123, criterion='gini', splitter='best')

dt_model = model_train(unrestricted_model, "UnRestricted Decision Tree")
predicted = dt_model.predict(X_train)

plot_roc_curve(dt_model, X_train, Y_train)

print(accuracy_score(Y_train, predicted))
random_forest = RandomForestClassifier(random_state=123, criterion='gini')

random_forest = model_train(random_forest, "Random Forest Classifier")
model_train(LogisticRegression(max_iter=100), "Logistic Regression")
model_train(LogisticRegression(max_iter=10000), "Logistic Regression")
model_train(LogisticRegression(max_iter=1000000), "Logistic Regression")
model_train(svm.SVC(kernel='linear'), 'SVM - Linear Kernel')
model_train(svm.SVC(kernel='poly'), 'SVM - Polynomial Kernel')
model_train(svm.SVC(kernel='rbf'), 'SVM - RBF Kernel')
model_train(svm.SVC(kernel='sigmoid'), 'SVM - Sigmoid Kernel')
model_train(GaussianNB(), 'Gaussian NB Model')
test = pd.read_csv('../input/titanic/train.csv')

test.head()
test.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)
test.head()
OHEncoder = OneHotEncoder()

embarked_columns_encoded = OHEncoder.fit_transform(test['Embarked'].values.reshape(-1, 1)).toarray()

embarked_coded = pd.DataFrame(embarked_columns_encoded, columns=["Embarked_" + str(int(i)) for i in range(embarked_columns_encoded.shape[1])])

test = test.join(embarked_coded)

test.head()
labelEncoder = LabelEncoder()

test['Sex'] = labelEncoder.fit_transform(test['Sex'])

test.head()
test = test.drop('Ticket', axis=1)
test.head()

test = test.drop('Embarked', axis=1)
test.head()
test.isnull().sum()
test.Age = test.groupby('Pclass')['Age'].apply(lambda x: x.fillna(x.mean()))
test.isnull().sum()
test[test.Fare.isnull()]
test.groupby('Pclass').Fare.describe()
df.head()
df.groupby('Pclass').Fare.describe()
test['Fare'].fillna(13.675550, inplace=True)
test_copy = pd.read_csv('../input/titanic/train.csv')

test_copy.head()
predicted = model.predict(test)
predicted_df = pd.DataFrame({'PassengerId': test_copy['PassengerId'], 'Survived': predicted})

predicted_df.head()
predicted_df.to_csv('final_submission.csv', index=False)