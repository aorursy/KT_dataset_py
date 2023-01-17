import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols

import scipy.stats as stats 
from scipy.stats.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

import seaborn as sns
wine=pd.read_csv("../input/winequality-red.csv")
wine.head()
wine.shape
wine.describe()#no missing values
wine.columns=['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality']

def hist_plotter(wine,column):
     wine[column].plot.hist(figsize=(10,5))
     plt.xlabel("Rating",fontsize=10)
     plt.ylabel(column,fontsize=10)
     plt.title(column+" vs Rating",fontsize=10)
     plt.show()
def skew(wine,col):
    wine[col] = np.log(wine[col])
    

hist_plotter(wine,'volatile_acidity')

skew(wine,'volatile_acidity')
hist_plotter(wine,'volatile_acidity')

hist_plotter(wine,'citric_acid')

skew(wine,'residual_sugar')
hist_plotter(wine,'residual_sugar')
skew(wine,'chlorides')
hist_plotter(wine,'chlorides')
skew(wine,'free_sulfur_dioxide')
hist_plotter(wine,'free_sulfur_dioxide')
skew(wine,'total_sulfur_dioxide')
hist_plotter(wine,'total_sulfur_dioxide')
hist_plotter(wine,'density')
hist_plotter(wine,'pH')
skew(wine,'sulphates')
hist_plotter(wine,'sulphates')
skew(wine,'alcohol')
hist_plotter(wine,'alcohol')
model=ols('chlorides ~ quality',data=wine).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print (aov_table) #take
model=ols('fixed_acidity ~ quality',data=wine).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print (aov_table) #take
model=ols('free_sulfur_dioxide ~ quality',data=wine).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print (aov_table) 
model=ols('citric_acid ~ quality',data=wine).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print (aov_table) #take
model=ols('residual_sugar ~ quality',data=wine).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print (aov_table) 
model=ols('total_sulfur_dioxide~ quality',data=wine).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print (aov_table) #take
model=ols('density ~ quality',data=wine).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print (aov_table) #take
model=ols('pH ~ quality',data=wine).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print (aov_table) 
model=ols('sulphates ~ quality',data=wine).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print (aov_table) #take
model=ols('alcohol ~ quality',data=wine).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print (aov_table) #take
model=ols('volatile_acidity ~ quality',data=wine).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print (aov_table) #take
y=wine['quality']
wine=wine.drop(['quality','pH','residual_sugar','free_sulfur_dioxide'],axis=1)
wine.head()
X_train, X_test, y_train, y_test = train_test_split(wine, y, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
clf = RandomForestClassifier(n_estimators=200)#how to set n_estimator
clf.fit(X_train, y_train)
pred=clf.predict(X_test)
print(classification_report(y_test, pred))
acc = clf.score(X_test,y_test)
acc
print(confusion_matrix(y_test, pred))
