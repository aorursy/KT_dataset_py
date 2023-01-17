import numpy as np
import pandas as pd
import tensorflow as tf
import keras

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set(palette="Set2")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,average_precision_score, confusion_matrix,
                             average_precision_score, precision_score, recall_score, roc_auc_score, )
from mlxtend.plotting import plot_confusion_matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


from xgboost import XGBClassifier, plot_importance
from imblearn.over_sampling import SMOTE
# read dataset
dataset = pd.read_csv("../input/Churn_Modelling.csv")
# first five row of the dataset
dataset.head()
dataset.describe()
# checking datatypes and null values
dataset.info()
dataset.drop(["RowNumber","CustomerId","Surname"], axis=1, inplace=True)
dataset.head()
_, ax = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.3)
sns.countplot(x = "NumOfProducts", hue="Exited", data = dataset, ax= ax[0])
sns.countplot(x = "HasCrCard", hue="Exited", data = dataset, ax = ax[1])
sns.countplot(x = "IsActiveMember", hue="Exited", data = dataset, ax = ax[2])
sns.pairplot(dataset)
encoder = LabelEncoder()
dataset["Geography"] = encoder.fit_transform(dataset["Geography"])
dataset["Gender"] = encoder.fit_transform(dataset["Gender"])
dataset["Age"].value_counts().plot.bar(figsize=(20,6))
import seaborn as sns

sns.set(rc={'figure.figsize':(11.7,8.27)})
facet = sns.FacetGrid(dataset, hue="Exited",aspect=3)
facet.map(sns.kdeplot,"Age",shade= True)
facet.set(xlim=(0, dataset["Age"].max()))
facet.add_legend()

plt.show()
import seaborn as sns

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.heatmap(dataset.corr(), annot=True)
_, ax =  plt.subplots(1, 2, figsize=(15, 7))
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
sns.scatterplot(x = "Age", y = "Balance", hue = "Exited", cmap = cmap, sizes = (10, 200), data = dataset, ax=ax[0])
sns.scatterplot(x = "Age", y = "CreditScore", hue = "Exited", cmap = cmap, sizes = (10, 200), data = dataset, ax=ax[1])
plt.figure(figsize=(8, 8))
sns.swarmplot(x = "HasCrCard", y = "Age", data = dataset, hue="Exited")
import seaborn as sns

sns.set(rc={'figure.figsize':(11.7,8.27)})
facet = sns.FacetGrid(dataset, hue="Exited",aspect=3)
facet.map(sns.kdeplot,"Balance",shade= True)
facet.set(xlim=(0, dataset["Balance"].max()))
facet.add_legend()

plt.show()
_, ax = plt.subplots(1, 2, figsize=(15, 6))
sns.scatterplot(x = "Balance", y = "Age", data = dataset, hue="Exited", ax = ax[0])
sns.scatterplot(x = "Balance", y = "CreditScore", data = dataset, hue="Exited", ax = ax[1])
facet = sns.FacetGrid(dataset, hue="Exited",aspect=3)
facet.map(sns.kdeplot,"CreditScore",shade= True)
facet.set(xlim=(0, dataset["CreditScore"].max()))
facet.add_legend()

plt.show()
plt.figure(figsize=(12,6))
bplot = dataset.boxplot(patch_artist=True)
plt.xticks(rotation=90)       
plt.show()
plt.subplots(figsize=(11,8))
sns.heatmap(dataset.corr(), annot=True, cmap="RdYlBu")
plt.show()
X = dataset.drop("Exited", axis=1)
y = dataset["Exited"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
import warnings
warnings.filterwarnings('ignore')
def initialize():
    global overfit_param , test_accuracy ,train_accuracy ,model , F1
    overfit_param=[]
    test_accuracy=[]
    train_accuracy=[]
    F1 =[]
    model=[]
def LR():
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(solver = 'liblinear' , random_state=0)#random_state=0, solver='lbfgs',multi_class='multinomial').fit(df_train, y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    accuracy = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv = 10)
    ta = accuracy.mean()
    accuracy = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv = 10)
    tsa = accuracy.mean()
    overfit_param.append(ta-tsa)
    test_accuracy.append(tsa)
    train_accuracy.append(ta)
    model.append('Logistic Regression')
    y_true=y_test
    from sklearn.metrics import f1_score
    F1.append(f1_score(y_true, y_pred, average='weighted'))
    print("----------Logistic Regression----------")

def SGD():
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    accuracy = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv = 10)
    ta = accuracy.mean()
    accuracy = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv = 10)
    tsa = accuracy.mean()
    overfit_param.append(ta-tsa)
    test_accuracy.append(tsa)
    train_accuracy.append(ta)
    model.append('SGDClassifier')
    y_true=y_test
    from sklearn.metrics import f1_score
    F1.append(f1_score(y_true, y_pred, average='weighted'))
    print("----------SGD----------")

def svm_scale():
    from sklearn import svm
    clf = svm.SVC(gamma='scale')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    accuracy = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv = 10)
    ta = accuracy.mean()
    accuracy = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv = 10)
    tsa = accuracy.mean()
    overfit_param.append(ta-tsa)
    test_accuracy.append(tsa)
    train_accuracy.append(ta)
    model.append('SVM - Gamma --> scale')
    y_true=y_test
    from sklearn.metrics import f1_score
    F1.append(f1_score(y_true, y_pred, average='weighted'))
    print("----------SVM - Gamma --> scale----------")

def svm_auto():
    from sklearn import svm
    clf = svm.SVC(gamma='auto')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    accuracy = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv = 10)
    ta = accuracy.mean()
    accuracy = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv = 10)
    tsa = accuracy.mean()
    overfit_param.append(ta-tsa)
    test_accuracy.append(tsa)
    train_accuracy.append(ta)
    model.append('SVM - Gamma --> auto')
    y_true=y_test
    from sklearn.metrics import f1_score
    F1.append(f1_score(y_true, y_pred, average='weighted'))
    print("----------SVM - Gamma --> auto----------")

def NuSVC():
    from sklearn.svm import NuSVC
    clf = NuSVC(gamma='scale')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    accuracy = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv = 10)
    ta = accuracy.mean()
    accuracy = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv = 10)
    tsa = accuracy.mean()
    overfit_param.append(ta-tsa)
    test_accuracy.append(tsa)
    train_accuracy.append(ta)
    model.append('NuSVM - Gamma --> scale')
    y_true=y_test
    from sklearn.metrics import f1_score
    F1.append(f1_score(y_true, y_pred, average='weighted'))
    print("----------NuSVC----------")

def LinearSVC():
    from sklearn.svm import LinearSVC
    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    accuracy = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv = 10)
    ta = accuracy.mean()
    accuracy = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv = 10)
    tsa = accuracy.mean()
    overfit_param.append(ta-tsa)
    test_accuracy.append(tsa)
    train_accuracy.append(ta)
    model.append('LinearSVM - Gamma --> scale')
    y_true=y_test
    from sklearn.metrics import f1_score
    F1.append(f1_score(y_true, y_pred, average='weighted'))
    print("----------LinearSVM - Gamma --> scale----------")

def KNN():
    from sklearn.neighbors import  KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    clf= KNeighborsClassifier(n_neighbors=2)
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    accuracy = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv = 10)
    ta = accuracy.mean()
    accuracy = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv = 10)
    tsa = accuracy.mean()
    overfit_param.append(ta-tsa)
    test_accuracy.append(tsa)
    train_accuracy.append(ta)
    model.append('KNeighborsClassifier')
    y_true=y_test
    from sklearn.metrics import f1_score
    F1.append(f1_score(y_true, y_pred, average='weighted'))
    print("----------KNN----------")


def Gaussian():
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    accuracy = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv = 10)
    ta = accuracy.mean()
    accuracy = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv = 10)
    tsa = accuracy.mean()
    overfit_param.append(ta-tsa)
    test_accuracy.append(tsa)
    train_accuracy.append(ta)
    model.append('GaussianNB')
    y_true=y_test
    from sklearn.metrics import f1_score
    F1.append(f1_score(y_true, y_pred, average='weighted'))
    print("----------Gaussian----------")


def Bernouli():
    from sklearn.naive_bayes import BernoulliNB
    clf = BernoulliNB()
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    accuracy = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv = 10)
    ta = accuracy.mean()
    accuracy = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv = 10)
    tsa = accuracy.mean()
    overfit_param.append(ta-tsa)
    test_accuracy.append(tsa)
    train_accuracy.append(ta)
    model.append('BernouliNB')
    y_true=y_test
    from sklearn.metrics import f1_score
    F1.append(f1_score(y_true, y_pred, average='weighted'))
    print("----------Bernouli----------")



def Tree():
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    accuracy = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv = 10)
    ta = accuracy.mean()
    accuracy = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv = 10)
    tsa = accuracy.mean()
    overfit_param.append(ta-tsa)
    test_accuracy.append(tsa)
    train_accuracy.append(ta)
    model.append('tree')
    y_true=y_test
    from sklearn.metrics import f1_score
    F1.append(f1_score(y_true, y_pred, average='weighted'))
    print("----------Tree----------")

def RandomForest():
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    accuracy = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv = 10)
    ta = accuracy.mean()
    accuracy = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv = 10)
    tsa = accuracy.mean()
    overfit_param.append(ta-tsa)
    test_accuracy.append(tsa)
    train_accuracy.append(ta)
    model.append('Random Forest')
    y_true=y_test
    from sklearn.metrics import f1_score
    F1.append(f1_score(y_true, y_pred, average='weighted'))
    print("----------Random Forest----------")

def Adaboost():
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    accuracy = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv = 10)
    ta = accuracy.mean()
    accuracy = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv = 10)
    tsa = accuracy.mean()
    overfit_param.append(ta-tsa)
    test_accuracy.append(tsa)
    train_accuracy.append(ta)
    model.append('Ada Boost')
    y_true=y_test
    from sklearn.metrics import f1_score
    F1.append(f1_score(y_true, y_pred, average='weighted'))
    print("----------Ada Boost----------")

def Gradient():
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    accuracy = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv = 10)
    ta = accuracy.mean()
    accuracy = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv = 10)
    tsa = accuracy.mean()
    overfit_param.append(ta-tsa)
    test_accuracy.append(tsa)
    train_accuracy.append(ta)
    model.append('Gradient Boosting')
    y_true=y_test
    from sklearn.metrics import f1_score
    F1.append(f1_score(y_true, y_pred, average='weighted'))
    print("----------Gradient Boosting----------")

def MLP():
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    accuracy = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv = 10)
    ta = accuracy.mean()
    accuracy = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv = 10)
    tsa = accuracy.mean()
    overfit_param.append(ta-tsa)
    test_accuracy.append(tsa)
    train_accuracy.append(ta)
    model.append('MLPClassifier')
    y_true=y_test
    from sklearn.metrics import f1_score
    F1.append(f1_score(y_true, y_pred, average='weighted'))
    print("----------MLP----------")

def Cat():
    from catboost import CatBoostClassifier
    clf = CatBoostClassifier(depth=8,iterations=30,learning_rate=0.1,verbose=False)
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    accuracy = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv = 10)
    ta = accuracy.mean()
    accuracy = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv = 10)
    tsa = accuracy.mean()
    overfit_param.append(ta-tsa)
    test_accuracy.append(tsa)
    train_accuracy.append(ta)
    model.append('Cat Classifier')
    y_true=y_test
    from sklearn.metrics import f1_score
    F1.append(f1_score(y_true, y_pred, average='weighted'))
    print("----------CAT----------")
    
def Light():
    from lightgbm import LGBMClassifier
    clf = LGBMClassifier( random_state=5)
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    accuracy = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv = 10)
    ta = accuracy.mean()
    accuracy = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv = 10)
    tsa = accuracy.mean()
    overfit_param.append(ta-tsa)
    test_accuracy.append(tsa)
    train_accuracy.append(ta)
    model.append('Light Classifier')
    y_true=y_test
    from sklearn.metrics import f1_score
    F1.append(f1_score(y_true, y_pred, average='weighted'))
    print("----------Light----------")
    
def XGB():
    from xgboost.sklearn import XGBClassifier
    clf = XGBClassifier(depth=8,iterations=30,learning_rate=0.1,verbose=False)
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    accuracy = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv = 10)
    ta = accuracy.mean()
    accuracy = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv = 10)
    tsa = accuracy.mean()
    overfit_param.append(ta-tsa)
    test_accuracy.append(tsa)
    train_accuracy.append(ta)
    model.append('XGB')
    y_true=y_test
    from sklearn.metrics import f1_score
    F1.append(f1_score(y_true, y_pred, average='weighted'))
    print("----------XGB----------")
def ML():
    initialize()
    LR()
    SGD()
    svm_scale()
    svm_auto()
    #/NuSVC()
    LinearSVC()
    KNN()
    Gaussian()
    Bernouli()
    Tree()
    RandomForest()
    Adaboost()
    Gradient()
    MLP()
    Cat()
    Light()
ML()
MLmod ={
    'Model Used' : model,
    'Overfir Quotient' :overfit_param,
    'Train Accuracy':train_accuracy,
    'Test Accuracy':test_accuracy,
    'F1 SCORE':F1
}

FinalAssesmentModel = pd.DataFrame(MLmod)
FinalAssesmentModel
plt.figure(figsize=(14,8))
ax = sns.lineplot(x="Model Used", y="F1 SCORE", data=FinalAssesmentModel ,marker='D' )
plt.xticks(rotation=30)
plt.figure(figsize=(14,8))
ax = sns.lineplot(x="Model Used", y="Overfir Quotient", data=FinalAssesmentModel ,marker='D' )
plt.xticks(rotation=30)
plt.figure(figsize=(14,8))
ax = sns.lineplot(x="Model Used", y="Test Accuracy", data=FinalAssesmentModel ,marker='D' )
plt.xticks(rotation=30)
plt.figure(figsize=(14,8))
ax = sns.lineplot(x="Model Used", y="Train Accuracy", data=FinalAssesmentModel ,marker='D' )
plt.xticks(rotation=30)
#standardizing the input feature
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X
X.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from keras import Sequential
from keras.layers import Dense
classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=10))
#Second  Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
#Fitting the data to the training dataset
history = classifier.fit(X_train,y_train, validation_split=0.13, epochs=150, batch_size=10, verbose=0)
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
eval_model=classifier.evaluate(X_train, y_train)
eval_model
y_pred=classifier.predict(X_test)
y_pred =(y_pred>0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
