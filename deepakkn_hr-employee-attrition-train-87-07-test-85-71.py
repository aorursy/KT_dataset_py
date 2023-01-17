# Necessary Library

%matplotlib inline

import pandas as pd

import numpy as np

import pandas_profiling

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB, BernoulliNB

from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, precision_score

from sklearn.metrics import roc_curve, precision_recall_curve, classification_report

from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt 

plt.rc("font", size=14)

import seaborn as sns

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)
#load the data to dataframe

data = pd.read_csv("../input/employee-attrition/HR-Employee-Attrition.csv")
pandas_profiling.ProfileReport(data)
data.duplicated().sum()
data.drop_duplicates(inplace = True)
data.isna().sum()
data['Attrition'].value_counts()
data["Attrition"].unique()
data["Attrition"].replace({"yes":1,"no":0}, inplace = True)
data["Gender"].replace({"male":1,"female":0}, inplace = True)
data["OverTime"].replace({"yes":1,"no":0}, inplace = True)
data.corr()


data.drop(labels=['EmployeeCount','EmployeeNumber','StockOptionLevel','StandardHours','Over18','MonthlyIncome','JobLevel','YearsSinceLastPromotion'],axis=1,inplace=True)
data.corr()

sns.heatmap(data.corr())
data.columns
cat_col = data.select_dtypes(exclude=np.number).columns

num_col = data.select_dtypes(include=np.number).columns

print(cat_col)

print(num_col)
#One hot encoding

encoded_cat_col = pd.get_dummies(data[cat_col])

encoded_cat_col

data_ready_model = pd.concat([data[num_col],encoded_cat_col], axis = 1)
label_encoder = LabelEncoder()

for i in cat_col:

    data[i] = label_encoder.fit_transform(data[i])
std_scale = StandardScaler().fit(data)

data_std = std_scale.transform(data)

minmax_scale = MinMaxScaler().fit_transform(data)
X = data.drop(columns="Attrition")

X.shape

y = data[["Attrition"]]

y.shape
std = StandardScaler()

X_std = std.fit_transform(X)
norm = MinMaxScaler()

X_norm = norm.fit_transform(X)
def roc_draw(X_test, y_test,logreg):

    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

    plt.figure()

    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic')

    plt.legend(loc="lower right")

    plt.savefig('Log_ROC')

    plt.show()
def model_fit(model, X, y, roc = False, conf = False, threshold = 0.5):

    train_X, test_X, train_y, test_y =  train_test_split(X, y, test_size = 0.3, random_state=1)

    print(np.array(np.unique(test_y, return_counts=True)).T)

    model.fit(train_X, train_y)

    train_pred = model.predict(train_X)

    print("Train Accuracy : ",accuracy_score(train_pred,train_y))

    print("Train Recall : ",recall_score(train_y, train_pred))

    print("Train Precision : ",precision_score(train_y, train_pred))

    test_pred = model.predict(test_X)

    print("Test Accuracy : ",accuracy_score(test_pred,test_y))

    print("Test Recall : ",recall_score(test_y,test_pred))

    print("Test Precision : ",precision_score(test_y,test_pred))

    if roc:

        roc_draw(test_X, test_y, model)

    if conf:

        print("Test Data Confusion Matrix")

        print(confusion_matrix(test_pred,test_y))

    print("After Tuning Threshold")

    test_pred_prob = model.predict_proba(test_X)

    predict_threshold_test = np.where(test_pred_prob[:,1]>threshold,1,0)

    print("Test Accuracy : ",accuracy_score(predict_threshold_test,test_y))

    print("Test Recall : ",recall_score(test_y, predict_threshold_test))

    print("Test Precision : ",precision_score(test_y, predict_threshold_test))

    if conf:

        print("Test Data Confusion Matrix")

        print(confusion_matrix(predict_threshold_test,test_y))

        print(classification_report(test_y, predict_threshold_test))

    return model.predict_proba(train_X), model.predict_proba(test_X)
logistic = LogisticRegression()

train_pred_prob, test_pred_prob = model_fit(logistic, X, y, roc = True, conf = True, threshold=0.3)
predict_threshold_test = np.where(test_pred_prob[:,1]>0.7,1,0)
np.where(logistic.predict_proba(X)[:,1]>0.5,1,0)
logistic.predict_proba(X)
model_fit(logistic, X_std, y)
model_fit(logistic, X_norm, y)