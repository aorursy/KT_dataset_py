# importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
# defining the dataset
datapath = '../input/mushroom/mushroom.csv'
df = pd.read_csv(datapath)
print(df.describe())
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.nunique())
print(df.info())
sb.countplot(x='Class', data=df)
# pre-processing the data
le = LabelEncoder()
for feature in df.columns :
    df[feature] = le.fit_transform(df[feature])
print(df.head())
plt.figure(figsize=(20, 15))
corr = df.corr()
sb.heatmap(corr, annot = True)
plt.show()
df = df.drop(["VeilType"],axis=1)
print('VeliType' in df.columns)
df_div = pd.melt(df, "Class", var_name="Characteristics")
fig, ax = plt.subplots(figsize=(20,15))
p = sb.violinplot(ax = ax, x="Characteristics", y="value", hue="Class", split = True, data=df_div, inner = 'quartile', palette = 'Set1')
df_no_class = df.drop(["Class"],axis = 1)
p.set_xticklabels(rotation = 90, labels = list(df_no_class.columns));
# Split into features and classes
x = df.loc[:, df.columns != "Class"]
y = df["Class"]
# splitting data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
# algorithm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
# model object
clfBayes = GaussianNB()
clfKNN = KNeighborsClassifier()
clfForest = RandomForestClassifier()
clfSVM = SVC()
clfLR = LogisticRegression()
clfTree = DecisionTreeClassifier()
clfNeural = MLPClassifier()
# fitting model in decision tree
clfTree = clfTree.fit(x_train,y_train)
from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(clfTree, out_file=None,
                         feature_names=x.columns,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graphviz.Source(dot_data)
features_list = x.columns.values
feature_importance = clfTree.feature_importances_
sorted_idx = np.argsort(feature_importance)


plt.figure(figsize=(20,7))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importances')
plt.draw()
plt.show()
drop_list = ['CapColor', 'Odor', 'GillAttachment', 'StalkRoot', 'GillColor', 'StalkSurfaceAboveRing', 'StalkColorAboveRing', 'StalkColorBelowRing', 'RingType' ,'CapShape']
for drop_feature in drop_list:
  df = df.drop([drop_feature],axis=1)
print(df.columns)
print(df.columns.size)
# accuracy metrix
models = []

# fitting model in bayes
clfBayes.fit(x_train,y_train)
pred = clfBayes.predict(x_test)
models.append(accuracy_score(y_test,pred))

# fitting model in knn
clfKNN.fit(x_train,y_train)
pred = clfKNN.predict(x_test)
models.append(accuracy_score(y_test,pred))

# fitting model in forest
clfForest.fit(x_train,y_train)
pred = clfForest.predict(x_test)
models.append(accuracy_score(y_test,pred))

# fitting model in bayes
clfSVM.fit(x_train,y_train)
pred = clfSVM.predict(x_test)
models.append(accuracy_score(y_test,pred))

# fitting model in Logistic regression
clfLR.fit(x_train,y_train)
pred = clfLR.predict(x_test)
models.append(accuracy_score(y_test,pred))


# fitting model in decision tree
clfTree.fit(x_train,y_train)
pred = clfTree.predict(x_test)
models.append(accuracy_score(y_test,pred))

# fitting model in neural network
clfNeural.fit(x_train,y_train)
pred = clfNeural.predict(x_test)
models.append(accuracy_score(y_test,pred))

# printing accuracy
print('Bayes accuracy = ', models[0])
print('KNN accuracy = ', models[1])
print('Random Forest accuracy = ', models[2])
print('SVM accuracy = ', models[3])
print('Logistic Regression accuracy = ', models[4])
print('Decision Tree accuracy = ', models[5])
print('Neural Network accuracy = ', models[6])

names = ['Bayes', 'KNN', 'RF', 'SVM', 'LR', 'DT', 'NN']
plt.figure(figsize=(20,5))
plt.bar(names, models, color=['black', 'red', 'green', 'blue', 'cyan', 'yellow', 'purple'])
plt.show()
y_pred = clfNeural.predict(x_test)
conf = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 6))
sb.heatmap(conf , annot = True,  linewidths=.5, cbar =None)
plt.title('Neural Network Classifier confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label');
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

plt.figure(figsize=(8,8))
plt.title('Reciever Operating Characteristics')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(fpr,tpr, color='purple')
plt.show()
plt.figure(figsize=(8,8))
plt.title('Area Under Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

auc = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="auc = "+str(auc), color='darkorange')
plt.legend()
plt.show()