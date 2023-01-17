import pandas as pd
df = pd.read_csv("/kaggle/input/titanic/train.csv")
df.head(10)
df = df.drop(['Name','Ticket','Cabin','Embarked','PassengerId'],axis=1)
df
df.isnull().sum()

df['Age'].fillna(df['Age'].median(),inplace=True)
df.head()
from sklearn.preprocessing import MinMaxScaler
df[['Age','Fare']] = MinMaxScaler().fit_transform(df[['Age','Fare']])

df = pd.get_dummies(df, prefix=['Sex'], columns=['Sex'])
df
X = df[['Pclass','Age','Fare','SibSp','Parch','Fare','Sex_female','Sex_male']]
X
y = df[['Survived']]
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape,y_train.shape)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_recall_fscore_support

DecisionTreemodel = DecisionTreeClassifier()
# Train Decision Tree Classifer
result = DecisionTreemodel.fit(X_train,y_train)
y_pred = DecisionTreemodel.predict(X_test)

precision_recall_fscore_tree = precision_recall_fscore_support(y_test, y_pred, average='macro')
print("Accuracy:",accuracy_score(y_test, y_pred))
print('precision_recall_fscore_tree:',precision_recall_fscore_tree)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
Gaussianmodel = GaussianNB()
result = Gaussianmodel.fit(X_train,y_train)
y_pred = Gaussianmodel.predict(X_test)
precision_recall_fscore_Gaussian = precision_recall_fscore_support(y_test, y_pred, average='macro')
print("Accuracy:",accuracy_score(y_test, y_pred))
print('precision_recall_fscore_Gaussian:',precision_recall_fscore_Gaussian)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
y_pred = clf.predict(X_test)
precision_recall_fscore_MLP = precision_recall_fscore_support(y_test, y_pred, average='macro')
print("Accuracy:",accuracy_score(y_test, y_pred))
print('precision_recall_fscore_MLP:',precision_recall_fscore_MLP)
# from sklearn.metrics import confusion_matrix
# def evaluate_model(model,X_train,X_test,y_train,y_test):
#   model.fit(X_train,y_train)
#   y_pred = model.predict(X_test)
#   tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
#   #Recall
#   recall = tp/(tp+fn)
#   #Precision
#   precision = tp / (tp+fp)
#   #F-Measure
#   f = 2*precision*recall/(precision+recall)
#   print(type(model).__name__)
#   print('Recall : {}'.format(recall))
#   print('Precision : {}'.format(precision))
#   print('F : {}'.format(f)) 
#   return recall,precision,f
import numpy as np
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
total_tree = []
total_Gaussian = []
total_mlp = []
for train_index, test_index in kf.split(X_train):

    X_train, X_test = X_train[train_index], X_test[test_index]
    y_train, y_test = y_train[train_index], y_test[test_index]
    y_train = y_train.reshape(len(y_train),)
    y_test = y_test.reshape(len(y_test),)
    
    d_tree_model = DecisionTreeClassifier()
    gaussian_model = GaussianNB()
    mlp_model = MLPClassifier(random_state=1, max_iter=300)
    recall1,precision1,f1_1 = evaluate_model(d_tree_model, X_train, y_train, X_test, y_test)
    recall2,precision2,f1_2 = evaluate_model(gaussian_model, X_train, y_train, X_test, y_test)
    recall3,precision3,f1_3 = evaluate_model(mlp_model, X_train, y_train, X_test, y_test)
    total_tree.append([recall1,precision1,f1_1])
    total_Gaussian.append([recall2,precision2,f1_2])
    total_mlp.append([recall3,precision3,f1_3])
#find mean total_tree ,total_Gaussian ,total_mlp 