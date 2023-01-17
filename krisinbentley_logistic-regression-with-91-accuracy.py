import numpy as np

import pandas as pd

import os

from matplotlib import pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

sns.set(style='white')

sns.set(style='whitegrid', color_codes=True)
# Read CSV data file into DataFrame

df = pd.read_csv("../input/Admission_Predict.csv")

df.head()
df.describe()
#rename columns

df.columns=['id','GRE','TOFEL','UniversityRating','SOP','LOR','CGPA','Research','p']

#training dataset and testing dataset

target = df['p']

features = df[['GRE','TOFEL','UniversityRating','SOP','LOR','CGPA','Research']]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
#

ur=X_train['UniversityRating'].value_counts()

plt.bar(ur.index, ur, alpha=0.8)

plt.title('Distribution of applicants from different levels of univerisity')

plt.grid(True, linestyle = "--")

plt.ylabel('Number of applicants')

plt.xlabel('University Ratings')
#scale our data, from 0 to 1

scaler = preprocessing.StandardScaler()

scaler.fit(X_train)

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
c=y_train.quantile(0.5)  

y_train_binary=pd.DataFrame({'p':y_train,'binary':0})

y_train_binary['binary'][y_train_binary['p']>c]=1

y_train=y_train_binary['binary']





y_test_binary=pd.DataFrame({'p':y_test,'binary':0})

y_test_binary['binary'][y_test_binary['p']>c]=1

y_test=y_test_binary['binary']
logreg = LogisticRegression()

logreg.fit(X_train, y_train)



y_test_predict = logreg.predict(X_test)



y_train_predict= logreg.predict(X_train)
logreg_score = (logreg.score(X_test, y_test))

print('Misclassification rate is %.2f %%' %(100-logreg_score*100))
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

import sklearn.metrics as metrics

probs = logreg.predict_proba(X_train)

preds = probs[:,1]

##Computing false and true positive rates

fpr, tpr, threshold = metrics.roc_curve(y_train, preds)

print(pd.DataFrame({'fpr': fpr,'Sensitivity-tpr':tpr,'cutoff':threshold}).iloc[20:25])

roc_auc = metrics.auc(fpr, tpr)



# PLOTTING

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([-0.01, 1.01])

plt.ylim([-0.01, 1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

roc_auc = metrics.auc(fpr, tpr)

roc_auc 