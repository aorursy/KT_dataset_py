#EDA and understanding feature importance
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
#AGE - Fifty percent of cervical cancer diagnoses occur in women ages 35 - 54, 

#and about 20% occur in women over 65 years of age. The median age of diagnosis is 48 years. 

#About 15% of women develop cervical cancer between the ages of 20 - 30. 

#Cervical cancer is extremely rare in women younger than age 20.
#Young women with early abnormal changes who do not have regular examinations are at high risk for localized cancer by the time they are age 40, and for invasive cancer by age 50.

#it remains much more prevalent in African-Americans -- whose death rates are twice as high as Caucasian women. 

#Hispanic American women have more than twice the risk of invasive cervical cancer as Caucasian women, also due to a lower rate of screening

#high poverty levels are linked with low screening rates.
#HIGH SEXUAL ACTIVITY Human papilloma virus (HPV) is the main risk factor for cervical cancer.

#Women most at risk for cervical cancer are those with a history of multiple sexual partners, sexual intercourse at age 17 years or younger, or both. 

#FAMILY HISTORY Women have a higher risk of cervical cancer if they have a first-degree relative (mother, sister) who has had cervical cancer. 

#strong association between cervical cancer and long-term use of oral contraception (OC).

#having many children increases the risk for developing cervical cancer, particularly in women infected with HPV.

#Smoking is associated with a higher risk for precancerous changes (dysplasia) in the cervix and for progression to invasive cervical cancer

#Women with weak immune systems, (such as those with HIV / AIDS), are more susceptible to acquiring HPV.
from pandas_profiling import ProfileReport

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn.model_selection import train_test_split

from fancyimpute import KNN

from sklearn.preprocessing import OrdinalEncoder
train1=pd.read_csv("/kaggle/input/cervical-cancer-risk-classification/kag_risk_factors_cervical_cancer.csv")
train2 = train1.replace('?', np.nan)

train2.isna().sum()
#creating profile report in Python



report = ProfileReport(train2, title='Profile Report', html={'style':{'full_width':True}})

report
train2=train2.drop(['STDs_Time_since_first_diagnosis','STDs_Time_since_last_diagnosis'], axis=1)
train2=train2.drop_duplicates()
tranum=train2.select_dtypes(include="number")

tranum.dtypes
tracat=train2.select_dtypes(include="object")



tracat1=tracat.apply(pd.to_numeric)

tracat1.dtypes
train=pd.concat([tracat1,tranum],axis=1,join="inner")
#impute missing values

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

train12 = pd.DataFrame(my_imputer.fit_transform(train))

train12.columns=train.columns

train12
y=train12['Biopsy']

X=train12.drop(['Biopsy'], axis=1)
#Best feature selection

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(10,'Score'))  #print 10 best features


# Feature Importance



from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()

model.fit(X,y)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()


#get correlations of each features in dataset

corrmat = train12.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(train12[top_corr_features].corr(),annot=True,cmap="RdYlGn")
