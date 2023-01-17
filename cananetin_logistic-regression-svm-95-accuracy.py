#import basic libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline
#read the file

df = pd.read_csv("../input/south-german-credit-updated/german_credit.csv")

df.head()
#check dataset's size

df.shape
df["credit_risk"].value_counts()
#Is there any duplicate value in the dataset?

df.duplicated().value_counts()
#Is there any null value in the dataset?

df.isna().sum()
#How many columns does this dataset have?

df.columns
df.info()
dataset = df.drop(["duration","amount","age"],1)

dataset.columns
from sklearn.preprocessing import LabelEncoder



for x in dataset.columns:

    dataset[x] = LabelEncoder().fit_transform(dataset[x])



dataset.head()
feature_x = dataset[['status', 'credit_history', 'purpose', 'savings',

       'employment_duration', 'installment_rate', 'personal_status_sex',

       'other_debtors', 'present_residence', 'property',

       'other_installment_plans', 'housing', 'number_credits', 'job', 'people_liable',

       'telephone', 'foreign_worker']]

feature_y = dataset[["credit_risk"]]
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif



def best_features(X_train,y_train,X_test):

    fs = SelectKBest(score_func = f_classif, k="all")

    fs.fit(X_train,y_train)

    X_train_fs = fs.transform(X_train)

    X_test_fs = fs.transform(X_test)

    return X_train_fs,X_test_fs,fs
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(feature_x,feature_y,test_size=0.33,random_state=21)

X_train_fs,X_test_fs,fs = best_features(X_train,np.ravel(y_train),X_test)
# what are scores for the features

for i in range(len(fs.scores_)):

	print('Feature %d: %f' % (i, fs.scores_[i]))

# plot the scores

plt.bar([i for i in range(len(fs.scores_))], fs.scores_)

plt.show()
train_x = dataset[['status', 'credit_history', 'purpose', 'savings','personal_status_sex',

              'property','other_installment_plans', 'housing','job','foreign_worker']]

train_y = dataset[["credit_risk"]]
X_train_lr,X_test_lr,y_train_lr,y_test_lr = train_test_split(train_x,train_y,test_size=0.33,random_state=21)
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(solver='liblinear')

fit = lr.fit(X_train_lr,np.ravel(y_train_lr))

fit
yhat = fit.predict(X_test_lr)

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test_lr, yhat)

print(confusion_matrix)
from sklearn.metrics import classification_report

print(classification_report(y_test_lr, yhat))
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_train_smote,y_train_smote = smote.fit_sample(train_x,train_y)
print("Before SMOTE: ", train_y["credit_risk"].value_counts())
print("After SMOTE: ", y_train_smote["credit_risk"].value_counts())
lr.fit(X_train_smote,np.ravel(y_train_smote))

y_pred2 = lr.predict(X_test_lr)

print(classification_report(y_test_lr, y_pred2))
from sklearn.svm import SVC

svclassifier = SVC(kernel="poly",degree=10)

svclassifier.fit(X_train_smote,np.ravel(y_train_smote))
yhat_svm = svclassifier.predict(X_test_lr)

print(classification_report(y_test_lr, yhat_svm))