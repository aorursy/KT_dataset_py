import numpy as np 

import pandas as pd 

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv("../input/diabetes.csv")

data.head()
data.shape
sns.countplot(x="Outcome", data= data)

plt.show()
X = data.drop("Outcome", axis = 1)

y = data["Outcome"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20, test_size=0.2)
from sklearn.linear_model import LogisticRegression



# instantiate the model (using the default parameters)

logreg = LogisticRegression()



# fit the model with data

logreg.fit(X_train,y_train)



# store the predictions

y_pred=logreg.predict(X_test)
from sklearn import metrics

cf_matrix = metrics.confusion_matrix(y_test, y_pred)

cf_matrix
sns.heatmap(pd.DataFrame(cf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))
y_pred_proba = logreg.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.plot([0, 1], [0, 1],'r--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.show()
from sklearn.feature_selection import RFE



# feature extraction

model = LogisticRegression()

rfe = RFE(model, 5)

fit = rfe.fit(X_train, y_train)

print("Num Features: ", fit.n_features_)

print("Selected Features: ",  fit.support_)

print("Feature Ranking: ", fit.ranking_)
X_train_f = X_train[["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction"]]

X_test_f  = X_test[["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction"]]



# store the predictions

logreg.fit(X_train_f,y_train)



# store the predictions

y_pred=logreg.predict(X_test_f)
cf_matrix = metrics.confusion_matrix(y_test, y_pred)

cf_matrix
sns.heatmap(pd.DataFrame(cf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))
y_pred_proba = logreg.predict_proba(X_test_f)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.plot([0, 1], [0, 1],'r--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.show()
#Grid Search

from sklearn.model_selection import GridSearchCV

clf = LogisticRegression()

grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}

grid_clf = GridSearchCV(clf, param_grid = grid_values,scoring = 'recall')

grid_clf.fit(X_train, y_train)



#Predict values based on new parameters

y_pred = grid_clf.predict(X_test)



# New Model Evaluation metrics 

print('Accuracy Score : ' + str(metrics.accuracy_score(y_test,y_pred)))

print('Precision Score : ' + str(metrics.precision_score(y_test,y_pred)))

print('Recall Score : ' + str(metrics.recall_score(y_test,y_pred)))
#Logistic Regression (Grid Search) Confusion matrix

cf_matrix = metrics.confusion_matrix(y_test,y_pred)

cf_matrix
sns.heatmap(pd.DataFrame(cf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)

auc = metrics.roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.plot([0, 1], [0, 1],'r--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.show()