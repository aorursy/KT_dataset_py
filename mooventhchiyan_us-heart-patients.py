# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing the neccasry Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Data Loading

df=pd.read_csv("/kaggle/input/heart-disease-prediction-using-logistic-regression/framingham.csv")

df.head()
df.TenYearCHD.value_counts()
df.TenYearCHD.value_counts(normalize=True).plot(kind='bar')

plt.show()
## Checking for Null Values and imputing
df.info()
df.isnull().sum()
# Forward and Backward fill is used to fill the null values so the distribution is not affected

df.fillna(method='ffill',inplace=True)

df.fillna(method='bfill',inplace=True)
df.info()
# Checking for outliers in the data set



cols=["age","cigsPerDay","totChol","sysBP","diaBP","BMI","heartRate","glucose"]



for col in cols:

    sns.boxplot(df[col])

    #df[col].plot(kind='box')

    plt.show()
sns.scatterplot(df["BMI"],df["glucose"],hue=df["TenYearCHD"])
import statsmodels.api as sm

y=df.TenYearCHD

X=df.drop('TenYearCHD',axis=1)
Xc=sm.add_constant(X)

model=sm.Logit(y,Xc)

result=model.fit()

result.summary()
# check multicollinearity

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])]

pd.DataFrame({'vif': vif[1:]}, index=X.columns).T
# VIF tabel shows tht there is no serious multicollonearity

# backward elimination to drop varables one by one



cols=list(Xc.columns)

p=[]

while len(cols)>2:

    Xc=Xc[cols]

    model=sm.Logit(y,Xc).fit().pvalues

    p=pd.Series(model.values[1:],index=Xc.columns[1:])

    pmax= max(p)

    pid=p.idxmax()

    

    if pmax>0.05:

        cols.remove(pid)

        print('column removed :',pid,pmax)

    else:

        break

        

cols
model=sm.Logit(y,Xc[cols])

result=model.fit()

result.summary()
# Checking the coeffecients of the features

exp_cof=np.exp(result.params)

exp_cof
## Age

## 1.Positive sign of age co efficient indicate that, probability of CHD increases with age

## 2.When age increase by 1 yr, log(odds) of CHD increase by 0.0646

## 3.When age increase by 1 yr, odds of CHD increase by 6%(So 1.066-1)
## Male

# 1.Positive sign of male co efficient indicate that, probability of CHD in male is high.

# 2.log(odds) of CHD for male is higher by 0.49 compared to female

# 3.odds(male)/odds(female)=1.63, odds(male) is 63% higher compared to odds(female)

## Assiging the threshold to determine the prediction from probability
prob=result.predict(Xc[cols])

prob.name='prob'

df_pred=pd.DataFrame([prob,y]).T

df_pred['pred']=df_pred['prob'].apply(lambda x:0 if x<0.5 else 1)

df_pred
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
confusion_matrix(df_pred.TenYearCHD,df_pred.pred)
accuracy_score(df_pred['TenYearCHD'],df_pred['pred'])
print(classification_report(df_pred['TenYearCHD'],df_pred['pred']))
from sklearn.metrics import roc_auc_score , roc_curve
print('AUC for model:',roc_auc_score(df_pred['TenYearCHD'],df_pred['prob']))

print('ROC for model:',roc_curve(df_pred['TenYearCHD'],df_pred['prob']))
fpr,tpr,threshold=roc_curve(df_pred['TenYearCHD'],df_pred['prob'])
plt.figure(figsize=(10,5))

plt.plot(fpr,tpr)

plt.plot(fpr,fpr,'r-')

plt.xlabel('fpr')

plt.ylabel('tpr')
threshold[0]=threshold[0]-1
plt.figure(figsize=(10,5))

plt.plot(fpr,tpr)

plt.plot(fpr,fpr,'r-')

plt.plot(fpr,threshold,'g-')

plt.xlabel('fpr')

plt.ylabel('tpr')
# Model is not performing its best in estimation, there is a large trade of bias and variance in the  model
# Declare x,y and split using X_train and y_train

X= df.drop(['TenYearCHD'],axis='columns')

y= df.TenYearCHD

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=1)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
# Determine the logisticRegression to validate x_test and y_test

# Logistic Regression is maximum likehood model-- iteration till the maximum

# Using solver we can stimulate and converge at a faster rate

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(fit_intercept=True,solver='liblinear')
lr.fit(X_train, y_train)  

# In stats we provide y,x while machine learning we provide X,y

# Check for any warning if its there again you need to do that
# Determine the prediction and Probability

y_train_prob = lr.predict_proba(X_train)[:,1]

y_train_pred = lr.predict(X_train)

y_train_prob  # output- Probability for 0 and 1 i.e ( P,(1-P)) # thats the reason we slice and take the value for 1 alone
# Evaluating on the training data

# Determine the confusion matrix ,accuracy_score,roc_curve,roc_auc_score,classification_report

from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,roc_auc_score,classification_report

print("confusion_matrix \n",confusion_matrix(y_train,y_train_pred))         # Evaluating the data on the trained data set-->Train and Predicted

print("accuracy_score",accuracy_score(y_train,y_train_pred))           # Train and Predicted

print("roc_accuracy acore",roc_auc_score(y_train,y_train_prob))                 # Train and Probability

print("classification_report \n ",classification_report(y_train,y_train_pred))   # Train and Predicted

fpr, tpr, thresholds =roc_curve(y_train,y_train_prob)  # Train and Probability-- Plotting



plt.plot(fpr, tpr)

plt.plot(fpr, fpr, "r-")

plt.show()
# Validating the data on the test data set

y_test_prob = lr.predict_proba(X_test)[:,1]

y_test_pred = lr.predict(X_test)

print("confusion_matrix \n",confusion_matrix(y_test,y_test_pred))         # Validating the data on the trained data set-->Train and Predicted

print("accuracy_score",accuracy_score(y_test,y_test_pred))           # Test and Predicted

print("roc_accuracy acore",roc_auc_score(y_test,y_test_prob))                 # Test and Probability

print("classification_report \n ",classification_report(y_train,y_train_pred))   # Test and Predicted

fpr,tpr,thresholds = roc_curve(y_test,y_test_prob)  # Test and Probability-- Plotting

plt.plot(fpr, tpr)

plt.plot(fpr, fpr, "r-")

plt.show()
# Model is Performing good in both Test and Training data Set
# IF we want to change the thresold to estimate the predicts( threshold= 0.25)



from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score



X, y = make_classification(

    n_classes=2, class_sep=1.5, weights=[0.9, 0.1],

    n_features=20, n_samples=1000, random_state=10

)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



clf = LogisticRegression(class_weight="balanced")

clf.fit(X_train, y_train)

THRESHOLD = 0.25

preds = np.where(clf.predict_proba(X_test)[:,1] > THRESHOLD, 1, 0)



pd.DataFrame(data=[accuracy_score(y_test, preds), recall_score(y_test, preds),

                   precision_score(y_test, preds), roc_auc_score(y_test, preds)], 

             index=["accuracy", "recall", "precision", "roc_auc_score"])

## Statistical Model, Machine Learning Model both performs with a normal accuracy of 85% 

## If the threshold value is been reduced to 25% rather than 50% the accuarcy value increases with the sacrifice of precision