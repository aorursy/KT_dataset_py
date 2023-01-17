import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
df = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.head()
df.dtypes
df.shape
# Convert Totalcharges to numerical

df.TotalCharges = pd.to_numeric(df.TotalCharges, errors = 'coerce')
# Check NULL value

df.isnull().sum()
# Replace NULL value as 0

df = df.fillna(value=0)
# Remove CustomerID

df.drop(['customerID'],axis=1,inplace=True)
#Convert all the Yes/No data to binary

columns_yes_no = ['Churn','Partner','Dependents','PhoneService','PaperlessBilling','OnlineSecurity','OnlineBackup','DeviceProtection','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']

for item in columns_yes_no:

    df[item].replace(to_replace = 'Yes',value=1,inplace=True)

    df[item].replace(to_replace = 'No',value=0,inplace=True)

df.head()
#Convert all the categorical data to binary

df = pd.get_dummies(df)

df.head()
df.corr()['Churn'].sort_values(ascending=False)
df.drop(['gender_Female','gender_Male','PhoneService','MultipleLines_No phone service'],axis=1,inplace=True)

df.head()
y = df['Churn'].values

dropped = df.drop(columns=['Churn'])

X = dropped
# Scale the data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=24)
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(solver='liblinear')

log_reg.fit(X_train, y_train)
log_reg.predict(X_test)
log_reg.score(X_test,y_test)
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train, y_train)
knn.predict(X_test)
knn.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier

randomforest= RandomForestClassifier()

randomforest.fit(X_train,y_train)
randomforest.predict(X_test)
randomforest.score(X_test,y_test)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score
# Calculate Accuarcy Scores

def cal_evaluation(classifier, cm):

    tn = cm[0][0]

    fp = cm[0][1]

    fn = cm[1][0]

    tp = cm[1][1]

    accuracy  = (tp + tn) / (tp + fp + fn + tn)

    precision = tp / (tp + fp)

    recall = tp / (tp + fn)

    print (classifier)

    print ("Accuracy is: %0.4f" % accuracy)

    print ("precision is: %0.4f" % precision)

    print ("recall is: %0.4f" % recall)



# Print Confusion Matrices

def draw_confusion_matrices(confusion_matricies):

    class_names = ['Stay','Churn']

    for cm in confusion_matrices:

        classifier, cm = cm[0], cm[1]

        cal_evaluation(classifier, cm)

        fig = plt.figure()

        ax = fig.add_subplot(111)

        cax = ax.matshow(cm, cmap=plt.get_cmap('Blues'))

        plt.title('Confusion matrix of %s' % classifier)

        fig.colorbar(cax)

        ax.set_xticklabels([''] + class_names)

        ax.set_yticklabels([''] + class_names)

        plt.xlabel('Predicted')

        plt.ylabel('Actually')

        plt.show()
confusion_matrices = [

    ("Logistic Regression", confusion_matrix(y_test,log_reg.predict(X_test))),

    ("K Nearest Neighbors", confusion_matrix(y_test,knn.predict(X_test))),

    ("Random Forest", confusion_matrix(y_test,randomforest.predict(X_test))),

    ]



draw_confusion_matrices(confusion_matrices)
from sklearn import metrics

from sklearn.metrics import roc_curve

probs = log_reg.predict_proba(X_test)

preds = probs[:, 1]

fpr_lr, tpr_lr, _ = roc_curve(y_test, preds)

plt.figure(1)

plt.plot([0, 1], [0, 1])

plt.plot(fpr_lr, tpr_lr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC curve - Logistic Regression')

plt.show()
metrics.auc(fpr_lr,tpr_lr)
probs = knn.predict_proba(X_test)

preds = probs[:, 1]

fpr_lr, tpr_lr, _ = roc_curve(y_test, preds)

plt.figure(1)

plt.plot([0, 1], [0, 1])

plt.plot(fpr_lr, tpr_lr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC curve - K Nearest Neighbors')

plt.show()
metrics.auc(fpr_lr,tpr_lr)
probs = randomforest.predict_proba(X_test)

preds = probs[:, 1]

fpr_lr, tpr_lr, _ = roc_curve(y_test, preds)

plt.figure(1)

plt.plot([0, 1], [0, 1])

plt.plot(fpr_lr, tpr_lr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC curve - Random Forest')

plt.show()
metrics.auc(fpr_lr,tpr_lr)
log_reg.coef_[0]

print ("Top 10 important attributes in our Logistic Regression Model")

for k,v in sorted(zip(map(lambda x: round(x, 3), log_reg.coef_[0]), \

                      dropped.columns), key=lambda k_v:(-abs(k_v[0]),k_v[1]))[0:10]:

    print (v + ": " + str(k))