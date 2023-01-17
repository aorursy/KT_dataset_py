import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
%matplotlib inline 
import matplotlib.pyplot as plt
df = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.head()
df.dtypes
df.isnull().sum()
df['SeniorCitizen'].value_counts()
df['MultipleLines'].value_counts()
df['TotalCharges'].value_counts()
df.replace(" ", np.nan, inplace = True)
df.dropna(subset=["TotalCharges"], axis=0, inplace = True)
df.reset_index(drop = True, inplace = True)
df.shape
cleanup_nums = {"PhoneService": {"Yes": 1, "No": 0}, "PaperlessBilling": {"Yes": 1, "No": 0}, "Churn": {"Yes": 1, "No": 0}}
df.replace(cleanup_nums, inplace=True)
df.head()
cleanup_nums = {"MultipleLines": {"Yes": 1, "No": 0, "No phone service": 2}}
df.replace(cleanup_nums, inplace=True)
df.head()
cleanup_nums = {"InternetService": {"Fiber optic": 1, "DSL": 2, "No": 0}}
df.replace(cleanup_nums, inplace=True)
df.head()
cleanup_nums = {"OnlineSecurity": {"Yes": 1, "No": 0, "No internet service": 2}, "DeviceProtection": {"Yes": 1, "No": 0, "No internet service": 2}, "TechSupport": {"Yes": 1, "No": 0, "No internet service": 2}, "StreamingTV": {"Yes": 1, "No": 0, "No internet service": 2}, "StreamingMovies": {"Yes": 1, "No": 0, "No internet service": 2}, "Contract": {"Month-to-month": 1, "Two year": 3, "One year": 2}, "PaymentMethod": {"Electronic check": 1, "Mailed check": 2, "Bank transfer (automatic)": 3, "Credit card (automatic)": 4}}
df.replace(cleanup_nums, inplace=True)
df.head()
cleanup_nums = {"OnlineBackup": {"Yes": 1, "No": 0, "No internet service": 2}}
df.replace(cleanup_nums, inplace=True)
df.head()
cleanup_nums = {"gender": {"Male": 1, "Female": 0}, "Partner": {"Yes": 1, "No": 0}, "Dependents": {"Yes": 1, "No": 0}}
df.replace(cleanup_nums, inplace=True)
df.head()
import seaborn as sns
plt.figure(figsize=(12,7))
sns.heatmap(cbar=True,annot=True,data=df.corr()*100,cmap='Greens')
plt.title('% Correlation Matrix')
plt.show()

X = np.asarray(df[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]])
X[0:5]
Y = np.asarray(df['Churn'])
Y [0:5]
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  Y_train.shape)
print ('Test set:', X_test.shape,  Y_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,Y_train)
LR
yhat = LR.predict(X_test)
yhat

yhat_prob = LR.predict_proba(X_test)
yhat_prob
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(Y_test, yhat)
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cnf_matrix = confusion_matrix(Y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= True,  title='Confusion matrix')
print (classification_report(Y_test, yhat))
from sklearn.metrics import log_loss
log_loss(Y_test, yhat_prob)
