import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import pandas as pd

import sklearn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import warnings

warnings.filterwarnings("ignore")



from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing
teste = pd.read_csv("../input/adult-pmr3508/test_data.csv")

treino = pd.read_csv("../input/adult-pmr3508/train_data.csv")

clear_teste = teste.dropna()

clear_treino = treino.dropna()
clear_treino.head()
clear_treino["income"] = clear_treino["income"].map({"<=50K": 0, ">50K":1})

clear_treino["sex"] = clear_treino["sex"].map({"Male": 0, "Female":1})
clear_treino.head()
sns.heatmap(clear_treino.corr(), annot=True, vmin=-1, vmax=1)
sns.lineplot('education.num', 'income', data=clear_treino)
sns.lineplot('hours.per.week', 'income', data=clear_treino)
sns.pairplot(clear_treino, hue='income')
names = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']

# Get column names first

# Create the Scaler object

scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object

x = clear_treino.loc[:, names].values

x = scaler.fit_transform(x)

scaled_df = pd.DataFrame(x, columns=names)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(scaled_df)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

data_treino_principal = pd.concat([principalDf, clear_treino[['income']]], axis = 1)

y = clear_treino.income
y
sns.pairplot(data_treino_principal, hue='income')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
log_pred = logmodel.predict(X_test)
log_pred
print("Logistic Regression Results")

print(confusion_matrix(y_test, log_pred))

print(classification_report(y_test, log_pred))

print('Accuracy: ',accuracy_score(y_test, log_pred))
from sklearn.svm import SVC

svm = SVC()

svm.fit(X_train,y_train)
svm_pred = svm.predict(X_test)
print("SVM metrics")

print(confusion_matrix(y_test, svm_pred))

print(classification_report(y_test, svm_pred))

print('Accuracy: ',accuracy_score(y_test, svm_pred))
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=0, max_depth=5)

decision_tree = decision_tree.fit(X_train, y_train)
decision_tree_pred= decision_tree.predict(X_test)
print('Accuracy: ',accuracy_score(y_test, decision_tree_pred))
from sklearn import linear_model

from sklearn.model_selection import cross_val_score





trainDS = pd.read_csv("../input/dataci/train.csv", names=["Id","latitude","longitude","median_age","total_rooms","total_bedrooms","population","households","median_income","median_house_value"], sep=r'\s*,\s*',engine='python',na_values='?', skiprows=1)

trainDS_X = trainDS[["Id","latitude","longitude","median_age","total_rooms","total_bedrooms","population","households","median_income"]]

trainDS_Y = trainDS[["median_house_value"]]





testDS_X = pd.read_csv("../input/dataci/test.csv", names=["Id","latitude","longitude","median_age","total_rooms","total_bedrooms","population","households","median_income"], sep=r'\s*,\s*',engine='python',na_values='?', skiprows=1)





plt.scatter(trainDS["latitude"],trainDS["median_house_value"])
regr = linear_model.LinearRegression()

regr.fit(trainDS_X,trainDS_Y)

testDS_Y = regr.predict(testDS_X)
print('Coefficients: \n', regr.coef_)



print('Interception: \n', regr.intercept_)
scores = cross_val_score(regr, trainDS_X, trainDS_Y, cv=10)

scores
regRCV = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100], cv=5)

regRCV.fit(trainDS_X,trainDS_Y)
regRCV.alpha_
regRCV = linear_model.RidgeCV(alphas=[6, 8, 10, 12, 14], cv=5)

regRCV.fit(trainDS_X,trainDS_Y)

regRCV.alpha_
regR = linear_model.Ridge(alpha = 10)
regR.fit(trainDS_X,trainDS_Y)
testDS_Y = regR.predict(testDS_X)

print('Coefficients: \n', regR.coef_)

print('Interception: \n', regR.intercept_)

scoresRidge = cross_val_score(regR, trainDS_X, trainDS_Y, cv=10)

scoresRidge

regLCV = linear_model.LassoCV(alphas=[0.1, 1, 10, 20, 100], cv=5)



regLCV.fit(trainDS_X,trainDS_Y["median_house_value"])

regLCV.alpha_

regLCV = linear_model.LassoCV(alphas=[15, 18, 20, 22, 24], cv=5)







regLCV.fit(trainDS_X,trainDS_Y["median_house_value"])





regLCV.alpha_
regL = linear_model.Lasso(alpha=22)





regL.fit(trainDS_X,trainDS_Y)





testDS_Y = regL.predict(testDS_X)





print('Coefficients: \n', regL.coef_)

print('Interception: \n', regL.intercept_)

scoresLasso = cross_val_score(regL, trainDS_X, trainDS_Y, cv=10)

scoresLasso



testDS_Predict = pd.DataFrame()



testDS_Predict["Id"] = testDS_X["Id"]

testDS_Predict["median_house_value"] = testDS_Y

testDS_Predict.head()




testDS_Predict.to_csv("testDS_3.csv", index=False)


