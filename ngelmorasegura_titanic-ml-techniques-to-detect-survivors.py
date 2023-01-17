import numpy as np

import pandas as pd

import seaborn as sns    # For graphical representation



import itertools

import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

import matplotlib.ticker as ticker



from sklearn import preprocessing

%matplotlib inline

import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



df_titanic = pd.read_csv('/kaggle/input/titanic/train.csv', error_bad_lines=False, engine="python", sep=",")

df_titanic.set_index("PassengerId", inplace = True)

print(df_titanic.shape)

print(df_titanic.ndim)
# There are three main methods of selecting columns in pandas:

#   1. using a dot notation, e.g. data.column_name,

#   2. using square braces and the name of the column as a string, e.g. data['column_name'] or

#   3. using numeric indexing and the iloc selector data.iloc[:, <column_number>]



# For selecting rows:

#   1. numeric row selection using the iloc selector, e.g. data.iloc[0:10, :] – select the first 10 rows.

#   2. label-based row selection using the loc selector 

#              (this is only applicably if you have set an “index” on your dataframe. e.g. data.loc[44, :]

#   3. logical-based row selection using evaluated statements, e.g. data[data["Area"] == "Ireland"]

#               – select the rows where Area value is ‘Ireland’.

df_titanic.iloc[0:3, :]
print(df_titanic.isna().sum())
print(df_titanic.dtypes)
print(df_titanic['Survived'].unique())
print(df_titanic['Sex'].unique())
dict_gender_map = {

  'male': 0,

  'female': 1

}



def gender_to_numeric(gender):

  return dict_gender_map[gender]



df_titanic['Sex'] = df_titanic['Sex'].apply(gender_to_numeric)
print(df_titanic['Embarked'].unique())
dict_embarked_map = {

  'S': 0,

  'C': 1,

  'Q': 2,

  np.nan: -1

}



def embarked_to_numeric(embarked):

  return dict_embarked_map[embarked]



df_titanic['Embarked'] = df_titanic['Embarked'].apply(embarked_to_numeric)
df_titanic['Age'].fillna(np.mean(df_titanic['Age'][:]), inplace = True)
# To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array:

X = df_titanic.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1).values

Y = df_titanic['Survived']
print(X[0:10])

print(Y[0:10])
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=4)

print ('Train set:', X_train.shape,  Y_train.shape)

print ('Test set:', X_test.shape,  Y_test.shape)
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, Y_train)

Y_LR_predicted = LR.predict(X_test)

Y_LR_predicted_prob = LR.predict_proba(X_test)

Y_LR_predicted[0:5]
print("Train set Accuracy: ", metrics.accuracy_score(Y_train, LR.predict(X_train)))

print("Test set Accuracy: ", metrics.accuracy_score(Y_test, Y_LR_predicted))
from sklearn.neighbors import KNeighborsClassifier
def bestK(max_number_of_Ks):

  mean_acc = np.zeros((max_number_of_Ks-1))

  std_acc = np.zeros((max_number_of_Ks-1))



  ConfusionMatrix = [];

  for n in range(1, max_number_of_Ks):

      neigh         = KNeighborsClassifier(n_neighbors = n).fit(X_train, Y_train)

      Y_predicted   = neigh.predict(X_test)

      mean_acc[n-1] = metrics.accuracy_score(Y_test, Y_predicted)

      std_acc[n-1]  = np.std(Y_predicted == Y_test)/np.sqrt(Y_predicted.shape[0])

  

  return (mean_acc.argmax() + 1, mean_acc.max())
k = bestK(25)[0]

print("k =", k)

# Train Model and Predict with the best K

neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, Y_train)

Y_bestK_predicted = neigh.predict(X_test)

Y_bestK_predicted[0:5]
print("Train set Accuracy: ", metrics.accuracy_score(Y_train, neigh.predict(X_train)))

print("Test set Accuracy: ", metrics.accuracy_score(Y_test, Y_bestK_predicted))
from sklearn.tree import DecisionTreeClassifier
def bestMax_Depth(max_number_of_Max_Depth):

  mean_acc = np.zeros((max_number_of_Max_Depth-1))

  std_acc = np.zeros((max_number_of_Max_Depth-1))



  ConfusionMatrix = [];

  for n in range(1, max_number_of_Max_Depth):

      tree          = DecisionTreeClassifier(criterion="entropy", max_depth = n).fit(X_train, Y_train)

      Y_predicted   = tree.predict(X_test)

      mean_acc[n-1] = metrics.accuracy_score(Y_test, Y_predicted)

      std_acc[n-1]  = np.std(Y_predicted == Y_test)/np.sqrt(Y_predicted.shape[0])

  

  return (mean_acc.argmax() + 1, mean_acc.max())
max_depth = bestMax_Depth(10)[0]

print("max_depth =", max_depth)

# Train Model and Predict with the best K

tree = DecisionTreeClassifier(criterion="gini", max_depth = max_depth).fit(X_train, Y_train)

Y_bestMax_Depth_predicted = tree.predict(X_test)

Y_bestMax_Depth_predicted[0:5]
print("Train set Accuracy: ", metrics.accuracy_score(Y_train, tree.predict(X_train)))

print("Test set Accuracy: ", metrics.accuracy_score(Y_test, Y_bestMax_Depth_predicted))
from sklearn import svm

svm_model = svm.SVC().fit(X_train, Y_train)

Y_svm_predicted = svm_model.predict(X_test)
print("Train set Accuracy: ", metrics.accuracy_score(Y_train, svm_model.predict(X_train)))

print("Test set Accuracy: ", metrics.accuracy_score(Y_test, Y_svm_predicted))
from sklearn.metrics import f1_score

from sklearn.metrics import log_loss
print("LogReg Accuracy train set:  %.4f" % metrics.accuracy_score(Y_train, LR.predict(X_train)))

print("LogReg Accuracy test set:   %.4f" % metrics.accuracy_score(Y_test, Y_LR_predicted))

print("LogReg F1-score test set:   %.4f" % f1_score(Y_test, Y_LR_predicted, average='weighted'))

print("LogReg LogLoss:             %.4f" % log_loss(Y_test, Y_LR_predicted_prob))



print("KNN Accuracy train set:  %.4f" % metrics.accuracy_score(Y_train, neigh.predict(X_train)))

print("KNN Accuracy test set:   %.4f" % metrics.accuracy_score(Y_test, Y_bestK_predicted))

print("KNN F1-score test set:   %.4f" % f1_score(Y_test, Y_bestK_predicted, average='weighted'))



print("DT Accuracy train set:   %.4f" % metrics.accuracy_score(Y_train, tree.predict(X_train)))

print("DT Accuracy test set:    %.4f" % metrics.accuracy_score(Y_test, Y_bestMax_Depth_predicted))

print("DT F1-score test set:    %.4f" % f1_score(Y_test, Y_bestMax_Depth_predicted, average='weighted'))



print("SVM Accuracy train set:  %.4f" % metrics.accuracy_score(Y_train, svm_model.predict(X_train)))

print("SVM Accuracy test set:   %.4f" % metrics.accuracy_score(Y_test, Y_svm_predicted))

print("SVM F1-score test set:   %.4f" % f1_score(Y_test, Y_svm_predicted, average='weighted'))
df_titanic_prediction = pd.read_csv('/kaggle/input/titanic/test.csv', error_bad_lines=False, engine="python", sep=",")

df_titanic_prediction.set_index("PassengerId", inplace = True)

print(df_titanic_prediction.shape)

print(df_titanic_prediction.ndim)
print(df_titanic_prediction.isna().sum())
df_titanic_prediction['Sex'] = df_titanic_prediction['Sex'].apply(gender_to_numeric)
df_titanic_prediction['Embarked'] = df_titanic_prediction['Embarked'].apply(embarked_to_numeric)
df_titanic_prediction['Age'].fillna(np.mean(df_titanic_prediction['Age'][:]), inplace = True)

df_titanic_prediction['Fare'].fillna(np.mean(df_titanic_prediction['Fare'][:]), inplace = True)
# To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array:

X_evaluation = df_titanic_prediction.drop(['Name', 'Ticket', 'Cabin'], axis=1).values

X_evaluation[0:10]
Y_bestMax_Depth_predicted_evaluation = tree.predict(X_evaluation)

Y_bestMax_Depth_predicted_evaluation[0:5]
submission = pd.DataFrame({\

                           'Survived': Y_bestMax_Depth_predicted_evaluation\

                          }, index = df_titanic_prediction.index)

submission.to_csv('decision_tree_submission.csv')

submission.head()