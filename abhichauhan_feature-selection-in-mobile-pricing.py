# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline 
import numpy as np
import pandas as pd
import sklearn
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from matplotlib.pyplot import figure 
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier , RandomForestClassifier  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression , Ridge ,Lasso , LogisticRegression , SGDClassifier
from sklearn.tree import DecisionTreeClassifier 
df = pd.read_csv('../input/train.csv')
df_copy = df.copy()
print(df_copy.columns)
df.head()
test_df =pd.read_csv('../input/test.csv')
print (test_df.info())
test_df = test_df.drop(['id'], axis = 1)
print (test_df.shape)


warnings.filterwarnings("ignore")

X = df.iloc[:,0:20]
y = df.iloc[:,-1]
# apply SelectKBest class to extract top 10 best features
best_features = SelectKBest(score_func = chi2 , k = 10)
fit = best_features.fit(X , y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)
f_score = pd.concat([df_scores , df_columns], axis = 1)
f_score.columns = ['score' , 'features']
print (f_score.nlargest(20 , 'score'))
# Feature importance of each feature
# the higher the score more important or relevant is the feature towards your output variable.
from sklearn.ensemble import ExtraTreesClassifier 
model = ExtraTreesClassifier()
model.fit(X , y)
# print (model.feature_importances_)
feature_imp = pd.Series(model.feature_importances_ , index = X.columns)
feature_imp.nlargest(20).plot(kind = 'barh')
figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')

plt.show()
# 3.Correlation Matrix with Heatmap
# Correlation can be positive (increase in one value of feature increases the value of the target variable) 
# or negative (increase in one value of feature decreases the value of the target variable)
# correlation mat of features
corrmat = df.corr()
feat = corrmat.index
print (feat)
plt.figure(figsize = (20,20))
g = sns.heatmap(df[feat].corr() , annot =True , cmap = 'RdYlGn')
sns.pointplot(data = df , y = 'int_memory', x = 'price_range')
plt.figure(figsize = (10 , 6))
df['fc'].hist(alpha = .5 , color = 'blue' , label = 'Front Cam')
df['pc'].hist(alpha = .5 , color = 'red' ,  label = 'Back Cam')
plt.legend()

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = .25 , train_size = .75)
def model(name,x):
    model_name = name()
    model_name.fit(X_train , y_train)
    print ("%s" %(x))
    print (model_name.score(X_test , y_test))
    print ("\n")

GBD = model(GradientBoostingClassifier,"GradientBoostingClassifier") 
RFC = model(RandomForestClassifier,"RandomForestClassifier") 
KNN = model(KNeighborsClassifier,"KNeighborsClassifier")
GBC = model(GaussianNB,"GaussianNB")
LR_ = model(LinearRegression,"LinearRegression")
Rid = model(Ridge,"Ridge")
Las = model(Lasso,"Lasso")
LRe = model(LogisticRegression,"LogisticRegression")
SGC = model(SGDClassifier,"SGDClassifier")
DTC = model(DecisionTreeClassifier,"DecisionTreeClassifier")
# removing irrelevant features
df_copy = df_copy.drop(['clock_speed','m_dep' , 'n_cores' , 'three_g' , 'four_g' ,'wifi'], axis = 1)
df_copy.head()

X = df_copy.iloc[:,0:14]
y = df_copy.iloc[:,-1]
print(X.columns)
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = .25 , train_size = .75)
print ("\n new model")
GBD = model(GradientBoostingClassifier,"GradientBoostingClassifier") 
RFC = model(RandomForestClassifier,"RandomForestClassifier") 
KNN = model(KNeighborsClassifier,"KNeighborsClassifier")
GBC = model(GaussianNB,"GaussianNB")
LR_ = model(LinearRegression,"LinearRegression")
Rid = model(Ridge,"Ridge")
Las = model(Lasso,"Lasso")
LRe = model(LogisticRegression,"LogisticRegression")
SGC = model(SGDClassifier,"SGDClassifier")
DTC = model(DecisionTreeClassifier,"DecisionTreeClassifier")
test_df = test_df.drop(['clock_speed','m_dep' , 'n_cores' , 'three_g' , 'four_g' ,'wifi'], axis = 1)
test_df.head()

# now we can choose any of the above models as per our wish , I prefer to choose GradientBoostClassifier
KNN = KNeighborsClassifier()
KNN.fit(X_train , y_train)
print (X_train.columns)
print (test_df.columns)
print (X_train.shape , test_df.shape)
predicted_price = KNN.predict(test_df)

test_df['price_range'] = predicted_price
test_df
