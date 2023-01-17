# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import libraries



import pandas as pd #working with dataframes

%matplotlib inline

import matplotlib.pyplot as plt #plotting

import seaborn as sns #plotting

from sklearn.preprocessing import StandardScaler, RobustScaler #scale data

from sklearn.model_selection import train_test_split #for splitting data

from imblearn.under_sampling import RandomUnderSampler #for undersampling 

from imblearn.over_sampling import RandomOverSampler #for oversampling

from imblearn.over_sampling import SMOTE #for smote

from imblearn.under_sampling import NearMiss  #near miss undersampling

from sklearn import tree # for decision tree

#for roc

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report # classification report: precision, recall

from sklearn.linear_model import LogisticRegression #linear regression

from sklearn.svm import SVC #svc

from sklearn.neighbors import KNeighborsClassifier #knn

from sklearn.ensemble import RandomForestClassifier #random forest

import xgboost as xgb #XGBoost
#load the data

dt=pd.read_csv("../input/creditcardfraud/creditcard.csv")

dt.head()
#to check for basic summary of the data

dt.describe().apply(lambda s: s.apply(lambda x: format(x, 'f')))
#check for missinf values

dt.isnull()

#check if we have any missing values

print(dt.isnull().values.any())

#check numver of total missing values

print(dt.isnull().sum())
print(dt['Class'].value_counts())

# proportion

print(dt['Class'].value_counts(normalize=True))
#plot

colors = ["#E43F5A", "#1B1B2F"]



sns.countplot('Class', data=dt, palette=colors)

plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)')


# fig=plt.figure()



amount_val = dt['Amount'].values

time_val = dt['Time'].values



# plt.subplot(2,1,1)

sns.distplot(amount_val, color='r')

plt.title('Distribution of Transaction Amount', fontsize=14)

plt.xlim([min(amount_val), max(amount_val)])





plt.show()





# plt.subplot(2,1,2)

sns.distplot(time_val,color='b')

plt.title('Distribution of Transaction Time', fontsize=14)

plt.xlim([min(time_val), max(time_val)])



plt.show()
plt.xticks([0,1])

sns.scatterplot(dt['Class'].values, dt['Amount'].values)

plt.title("class vs Amount")

plt.xlabel("Class")

plt.ylabel("Amount")

plt.show()
# Compute the correlation matrix

corr = dt.corr()



# Set up the matplotlib figure

fig = plt.figure(figsize = (12, 9)) 



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, vmax=.9, center=0, square=True)
#drop the time column



dt=dt.drop("Time",axis=1)

#can also be done by reassigning dt as dt.iloc(:,2:)

dt.head()
#convering class in categories

dt["Class"] = dt["Class"].astype('category')

dt["Class"] = dt["Class"].cat.rename_categories({0: 'Not_Fraud', 1: 'Fraud'})

dt["Class"]
scaler = RobustScaler().fit(dt.iloc[:,:-1])



scaler.transform(dt.iloc[:,:-1])

dt.head()
#split

x=dt.iloc[:,:-1]

y=dt.iloc[:,-1]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

#check proportion of class



print(yTrain.value_counts())

print(yTest.value_counts())

# proportion

print(yTrain.value_counts(normalize=True))

print(yTest.value_counts(normalize=True))
rus = RandomUnderSampler(sampling_strategy='auto',random_state=9650)

X_res, y_res = rus.fit_resample(xTrain, yTrain)

# print(X_res.value_counts())

print(y_res.value_counts())

# y_res.head()

print(X_res.shape)

print(y_res.shape)
ros = RandomOverSampler(sampling_strategy='auto',random_state=9650)

X_ros, y_ros = ros.fit_resample(xTrain, yTrain)

print(y_ros.value_counts())

print(X_ros.shape)

print(y_ros.shape)
sm = SMOTE(sampling_strategy='auto',random_state=9650)

X_sm, y_sm = sm.fit_resample(xTrain, yTrain)

print(y_sm.value_counts())

print(X_sm.shape)

print(y_sm.shape)
nr = NearMiss(sampling_strategy='auto')

X_nr, y_nr = nr.fit_resample(xTrain, yTrain)

print(y_nr.value_counts())

print(X_nr.shape)

print(y_nr.shape)
#function to plot ROC

def roc_plot(fpr,tpr):

    plt.plot(fpr, tpr, color='red', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.title("Receiver Operating Characteristic (ROC) Curve")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.legend()

    plt.show()

def test_auc_roc_classification_score(clf):

    probs = clf.predict_proba(xTest)

    probs = probs[:, 1]

    auc = roc_auc_score(yTest, probs)    

    print('AUC: %.2f' % auc)

    fpr, tpr, thresholds = roc_curve(yTest,probs, pos_label='Not_Fraud')

    roc_plot(fpr,tpr)

    predicted=clf.predict(xTest)

    report = classification_report(yTest, predicted)

    print(report)

    return auc
clf = tree.DecisionTreeClassifier()

#for original training dataset

clf = clf.fit(xTrain, yTrain)



test_auc_roc_classification_score(clf)
clf_under = tree.DecisionTreeClassifier()

clf_under = clf_under.fit(X_res, y_res)

test_auc_roc_classification_score(clf_under)
clf_o = tree.DecisionTreeClassifier()

clf_o = clf.fit(X_res, y_res)

test_auc_roc_classification_score(clf_o)
clf_sm = tree.DecisionTreeClassifier()

clf_sm = clf.fit(X_sm, y_sm)

test_auc_roc_classification_score(clf_sm)
clf_nr = tree.DecisionTreeClassifier()

clf_nr = clf.fit(X_nr, y_nr)

auc_nm=test_auc_roc_classification_score(clf_nr)
reg = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, multi_class='ovr').fit(X_ros, y_ros)

auc_reg=test_auc_roc_classification_score(reg)
knn_clf = KNeighborsClassifier().fit(X_ros, y_ros)

auc_knn=test_auc_roc_classification_score(knn_clf)
rf_clf = RandomForestClassifier().fit(X_ros, y_ros)

auc_rf=test_auc_roc_classification_score(rf_clf)
xgb_clf = xgb.XGBClassifier(max_depth=3, n_estimator=300, learning_rate=0.05).fit(X_ros, y_ros)

auc_xgb=test_auc_roc_classification_score(xgb_clf)
# print(auc_reg)

m=max(auc_xgb, auc_reg, auc_knn, auc_rf)

if(m==auc_xgb):

    print("XGBoost performs the best")

elif(m==auc_reg):

    print("Logistic Regresion performs the best")

elif(m==auc_rf):

    print("Random Forest performs the best")

else:

    print("KNN performs the best")