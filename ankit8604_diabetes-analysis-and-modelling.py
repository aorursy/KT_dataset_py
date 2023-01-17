# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# importing libraries

# imports 

import pandas as pd

import numpy as np

np.random.seed(42)

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,VotingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression,SGDClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.cross_validation import train_test_split

from IPython.display import Image

from sklearn.metrics.cluster import fowlkes_mallows_score

from sklearn.metrics import roc_curve,accuracy_score,auc,roc_auc_score,confusion_matrix,precision_score,recall_score,f1_score

from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import cross_val_score,cross_validate,cross_val_predict

from sklearn.preprocessing import StandardScaler
# reading file

dataframe = pd.read_csv('../input/diabetes.csv')

dataframe.head()
# dataframe['Age'].hist()

plt.hist(dataframe['Age'][dataframe['Outcome'] == 1],label="Patient vs Age")
sns.heatmap(dataframe.corr(),annot=True)
print(dataframe.isnull().sum())

print("Minimum Bloodpressure",dataframe['BloodPressure'].min())

print("Minimum BMI",dataframe['BMI'].min())
dataframe['BMI'] = dataframe['BMI'].replace(0,dataframe['BMI'].mean())

dataframe['BloodPressure'] = dataframe['BloodPressure'].replace(0,dataframe['BloodPressure'].mean())
dataframe['BloodPressure'].min()
Y = dataframe.Outcome

X = dataframe.drop('Outcome',axis=1)
# Now without feauture enginnering and data normalization lets check how our model performs on test and train data

classifier = SVC()

HPoptimizerSVC = GridSearchCV(classifier,param_grid={'C': [1,10],'gamma': [0.0001,0.001,0.01,0.1]})

classifiers = {'Random_Forest':RandomForestClassifier(),

               'Logistic_Reggression':LogisticRegression(),

               'Decision Tree Classifier' : DecisionTreeClassifier(),

               'SGDClassifier':SGDClassifier(),

               'naive_bayes':GaussianNB(),

               "Support_vector_Machine": HPoptimizerSVC,

               "AdaBoost" : AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=10),

               "ExtraForestClassifier" : GradientBoostingClassifier(), 

               'Multilayer Perceptron' : MLPClassifier(hidden_layer_sizes=(100,),momentum=0.9,solver='sgd'),

               'Voting Classifier' : VotingClassifier(estimators=[('log',LogisticRegression()),('SVM',SVC(C=1000)),('MLP',MLPClassifier(hidden_layer_sizes=(100,)))],voting='hard')



              }

#Holds accuracy for various models

Acc= {}

Acc_Train = {}

Acc_Test = {}

Predictions = {}

ROC = {}

AUC = {}

Confusion_Matrix = {}

Gmean = {}

Precision = {}

Recall = {}

F1_score = {}

mats = pd.DataFrame(Confusion_Matrix)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

scaler = StandardScaler()

X = scaler.fit_transform(X)
for clf in classifiers:

    Acc[clf] = cross_validate(classifiers[clf],X,Y,cv=10,n_jobs=-1,scoring='accuracy',return_train_score=True)

    Acc_Train[clf] =  Acc[clf]['train_score'].mean()

    Acc_Test[clf] = Acc[clf]['test_score'].mean()

    classifiers[clf].fit(scaler.transform(X_train),y_train)

    pred =  classifiers[clf].predict(scaler.transform(X_test))

    ROC[clf] = roc_auc_score(y_test,pred)

    AUC[clf] = auc(y_test,pred,reorder=True)

    Confusion_Matrix[clf] = confusion_matrix(y_test,pred)

    Gmean[clf] = fowlkes_mallows_score(y_test,pred)

    Precision[clf] = precision_score(y_test,pred)

    Recall[clf] = recall_score(y_test,pred)    

    F1_score[clf] = f1_score(y_test,pred)
Accuracy_train = pd.DataFrame([Acc_Train[vals]*100 for vals in Acc_Train],columns=['Accuracy_Train'],index=[vals for vals in Acc_Train])

Accuracy_pred = pd.DataFrame([Acc_Test[vals]*100 for vals in Acc_Test],columns=['Accuracy_Test'],index=[vals for vals in Acc_Test])
ROC_Area = pd.DataFrame([ROC[vals] for vals in ROC],columns=['ROC(area)'],index=[vals for vals in ROC])

AUC_Area = pd.DataFrame([AUC[vals] for vals in AUC],columns=['AUC(area)'],index=[vals for vals in AUC])

Gmean = pd.DataFrame([Gmean[vals] for vals in Gmean],columns=['Gmean'],index=[vals for vals in Gmean])

Prec = pd.DataFrame([Precision[vals] for vals in Precision],columns=['precision'],index=[vals for vals in Precision])

Rec = pd.DataFrame([Recall[vals] for vals in Recall],columns=['recall'],index=[vals for vals in Recall])

Prec = pd.DataFrame([Precision[vals] for vals in Precision],columns=['precision'],index=[vals for vals in Precision])

f1 =  pd.DataFrame([F1_score[vals] for vals in F1_score],columns=['f1_score'],index=[vals for vals in F1_score])
pd.concat([Accuracy_train,Accuracy_pred,ROC_Area,AUC_Area,Gmean,Prec,Rec,f1], axis=1)
CF = {}

for mat in Confusion_Matrix:

    CF[mat] = Confusion_Matrix[mat]

sns.heatmap(CF['Logistic_Reggression'],annot=True)    