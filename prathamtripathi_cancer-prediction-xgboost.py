import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

import matplotlib.pyplot as plt

import seaborn as sn



%matplotlib inline
df = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")

df = df.drop(['Unnamed: 32'], axis=1)

df.head()
df.dtypes
df.isnull().sum()
df["diagnosis"].unique()
le = LabelEncoder()

le.fit(["M","B"])

df["diagnosis"] = le.transform(df["diagnosis"])
df.head()
X = df[df.columns[df.columns!="diagnosis"]]

X.head()
y = df["diagnosis"]

y.head()
dmatrix = xgb.DMatrix(data = X,label = y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 123, stratify = y)
xgb_css = xgb.XGBClassifier(n_estimators = 100,objective = "reg:logistic",colsample_bytree = 0.3,learning_rate = 0.1, max_depth = 5,alpha =10)
xgb_css.fit(X_train,y_train)
pred = xgb_css.predict(X_test)
#Evaluation

from sklearn.metrics import confusion_matrix,classification_report

import itertools

def plot_confusion_matrix(cm,classes,

                         normalize = False,

                         title='Confusion Matrix',

                         cmap = plt.cm.Blues):

    if normalize:

        cm = cm.astype('float')/cm.sum(axis = 1)[:,np.newaxis]

        print("After Normalization")

    else:

        print("Without Normalization")

    print(cm)

    plt.imshow(cm,interpolation='nearest',cmap = 'Wistia')

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks,classes,rotation = True,color='white')

    plt.yticks(tick_marks,classes,rotation =True,color='white')

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max()/2

    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):

        plt.text(j,i,format(cm[i,j],fmt),

                horizontalalignment = "center",

                color = 'white' if cm[i,j]>thresh else "black")

        

    plt.tight_layout()

    plt.xlabel("Predicted",color='white',size=20)

    plt.ylabel("True",color='white',size=20)
cnf_matrix=confusion_matrix(y_test,pred,labels=[0,1])

np.set_printoptions(precision = 2)

plt.figure()

plot_confusion_matrix(cnf_matrix,classes=['benign(0)','malignant(1)'],normalize=False,title='Confusion Matrix')
print(classification_report(y_test,pred))
from sklearn.metrics import f1_score

f1_score(y_test,pred,average='weighted')
params = {"objective":"reg:logistic","colsample_bytree":"0.3","learning_rate": "0.1","max_depth":"5","alpha":"10"}

cv_results = xgb.cv(dtrain = dmatrix, params = params, nfold = 3,early_stopping_rounds =10,metrics="error", as_pandas = True, seed = 123)
cv_results.head()
print((cv_results["test-error-mean"]).tail(1))