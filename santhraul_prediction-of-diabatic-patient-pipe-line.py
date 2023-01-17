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
# supress warnings

import warnings

warnings.filterwarnings('ignore')
# read data

data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")

data.head()
# creating duplicate dataset

df = data
# visualise data with pairplot

from matplotlib import pyplot as plt

import seaborn as sns

sns.pairplot(df,hue="Outcome",diag_kind='kde')

plt.show()
#Feature Selection

# define X and Y variabel

X=df.drop('Outcome',axis=1)

y=df[['Outcome']]



# get feature importance using gradient boosting

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()

gb.fit(X,y)

gb.feature_importances_
# check for Feature importance

Feature_Importance_GB = pd.DataFrame({"Feature_Importance_GB" : gb.feature_importances_}, index=df.columns[:-1])

Feature_Importance_GB.sort_values(by = 'Feature_Importance_GB', ascending = False)
# Lets check the feature importance using SelectKBest (based on Pvalue)

from sklearn.feature_selection import SelectKBest,f_classif

skb = SelectKBest(f_classif,8)

skb.fit_transform(X,y)







# check for Feature importance

Kbest = pd.DataFrame({"Pvalue" : skb.pvalues_}, index=df.columns[:-1])

Kbest.sort_values(by = 'Pvalue', ascending = False)
# check for the desinilarity in the data set by ttest

plasdiabetic = df[df['Outcome']==1]['Glucose']

plasnondiabetic = df[df['Outcome']==0]['Glucose']

import scipy.stats as stats

print(stats.ttest_ind(plasdiabetic,plasnondiabetic))
df.info()
# fit and validate Zero model using Logistic regression in pipeline



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import PowerTransformer



#train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=20)



#define Pipeline

from sklearn.pipeline import Pipeline

pipe = Pipeline((

("pt",PowerTransformer()),

("sc",StandardScaler()),    

("lr", LogisticRegression()),

))

pipe.fit(X_train,y_train)



print("Testing Accuracy : ", pipe.score(X_test,y_test))

print("Training Accuracy: ", pipe.score(X_train,y_train))

#Pipeline Intermediate Step

pipe.named_steps['lr'].coef_
# check for basic ststistics in data set

df.describe()
# replacing zero values with np.nan

df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

df['Glucose'].replace(0,np.nan, inplace = True)

df['BloodPressure'].replace(0,np.nan, inplace = True)

df['Insulin'].replace(0,np.nan, inplace = True)

df['BloodPressure'].replace(0,np.nan, inplace = True)

df['SkinThickness'].replace(0,np.nan, inplace = True)
# check for info and null values

df.info()
# train test split

X=df.drop('Outcome',axis=1)

y=df[['Outcome']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=20)
#With Pipeline

from sklearn.pipeline import Pipeline



pipe = Pipeline((

("it", IterativeImputer()),

("pt",PowerTransformer()),

("sc", StandardScaler()),

("lr", LogisticRegression()),

))

pipe.fit(X_train,y_train)



print("Testing Accuracy : ", pipe.score(X_test,y_test))

print("Training Accuracy: ", pipe.score(X_train,y_train))
#Including SelectKBest

#With Pipeline

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

pipe = Pipeline((

("it", IterativeImputer()),

("pt",PowerTransformer()),

("sc", StandardScaler()),

("skb",SelectKBest(f_classif,k=3)),

("lr", LogisticRegression()),

))

pipe.fit(X_train,y_train)

print("Testing Accuracy : ", pipe.score(X_test,y_test))

print("Training Accuracy : ",pipe.score(X_train,y_train))
#Including RFE

#With Pipeline

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE



pipe = Pipeline((

("it", IterativeImputer()),

("pt",PowerTransformer()),

("sc", StandardScaler()),

("fs",RFE(estimator = LogisticRegression(),n_features_to_select=3, step=1)),

("lr", LogisticRegression()),

))

pipe.fit(X_train,y_train)



print("Testing Accuracy : ", pipe.score(X_test,y_test))

print("Training Accuracy: ", pipe.score(X_train,y_train))
pipe.named_steps['lr'].coef_
# check for clssification report

predicted = pipe.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report,recall_score,precision_score,f1_score

print(confusion_matrix(y_test,predicted))

print(classification_report(y_test,predicted))
print(recall_score(y_test,predicted,average=None))

print(precision_score(y_test,predicted,average=None))

print(f1_score(y_test,predicted,average=None))
#Evaluating models using Cross Validation

from sklearn.model_selection import cross_val_score

scoreslr = cross_val_score(pipe, X_train, y_train, cv=10, scoring='accuracy')

print(scoreslr)
# check for mean accuracy and Standard deviation

import numpy as np

print("Average Accuracy of my model: ", np.mean(scoreslr))

print("SD of accuracy of the model : ", np.std(scoreslr,ddof=1))
# 95% Confidence Interval of Accuracy

import scipy.stats

xbar = np.mean(scoreslr)

n=10

s = np.std(scoreslr,ddof=1)

se = s/np.sqrt(n)

stats.t.interval(0.95,df=n-1,loc=xbar,scale=se)