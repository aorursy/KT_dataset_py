import pandas as pd
import numpy as np
df = pd.read_csv('bank-marketing.csv')
df
df2 = df.drop(df[df['education'] == 'unknown'].index, axis = 0, inplace = False)
df2
from scipy.stats import zscore
print(df2['balance'].mean())
df2['baloutliers']= zscore(df2['balance'])
cle = (df2['baloutliers']>3) | (df2['baloutliers']<-3 )
df3 = df2.drop(df2[cle].index, axis = 0, inplace = False)
df4 = df3.drop('baloutliers', axis=1)
df5 = df4.drop('contact', axis=1)
df5['Month'] = df5['month']
Month = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
df5['Month'] = [Month[item] for item in df5['Month']]
df5
df5['duration'] = df5['duration'].apply(lambda n:n/60).round(2)
df6 = df5.drop(df5[df5['duration']<5/60].index, axis = 0, inplace = False)
df6
df7 = df6.drop(df6[df6['poutcome'] == 'other'].index, axis = 0, inplace = False)
df7
df7['Response'] = df7['response']
df7['Response'] = pd.get_dummies(df7['Response'], drop_first = True)
df['pdays'].describe()
ddf = df.copy()
ddf.drop(ddf[ddf['pdays'] == -1].index, inplace = True)
ddf['pdays'].describe()
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
%matplotlib inline
ddf2 = df7.copy()
ddf2['Edu'] = df7['education']
Education = {"primary":1,"secondary":2,"tertiary":3}
ddf2['Edu'] = [Education[item] for item in ddf2['Edu']]
barG = ddf2[['Edu','balance']].groupby("Edu").median().plot(kind='barh',legend = False,color = 'yellowgreen')
barG.set_ylabel("Education  \n1:primary , 2:secondary ,3: tertiary")
barG.set_xlabel("balance")
plt.show()
#ddf2.groupby('Edu').median()
#if someone wants to se numbers.
sns.boxplot(df7['pdays'])
print('outliers')
sns.catplot("response","duration",data = df7)
sns.catplot("response","balance",data = df7)
sns.catplot("response","pdays",data = df7)
sns.catplot("response","previous",data = df7) 
sns.catplot("response","campaign",data = df7)
g= sns.pairplot(df7)
plt.figure(figsize=(30,30))
ax = sns.heatmap(df7.corr(), annot = True, linewidth = 3)
ax.tick_params(size = 10, labelsize = 10)
plt.title("bank marketing", fontsize = 25)
plt.show()
df7.drop(['marital'],axis=1, inplace=True)
df8 = df7.iloc[:, 0:7]
df7.drop(['month'],axis=1, inplace=True)
df7.drop(['response'],axis=1, inplace=True)
df7 = pd.get_dummies(df7,drop_first=True)
df10=df7['Response'].copy()
df7.drop(['Response'],axis=1, inplace=True)
df7 = pd.merge(df7, df10, left_index = True, right_index = True)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
import warnings
warnings.filterwarnings('ignore')
models = []
models.append(('LR', LogisticRegression()))
df_train, df_test = train_test_split(df7, test_size=0.2, random_state=51)
X_train = df_train.drop('Response', axis=1)
y_train = df_train['Response']
 
print('Shape of X = ', X_train.shape)
print('Shape of y = ', y_train.shape) 
from sklearn.feature_selection import RFE
LR.fit(X_train, y_train)

rfe = RFE(LR, 10)  
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]
col
X_train_rfe = X_train[col]
import statsmodels.api as sm  
X_train_rfe = sm.add_constant(X_train_rfe)
LR = sm.OLS(y_train,X_train_rfe).fit()
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
import statsmodels.api as sm  
X_train_lm = sm.add_constant(X_train_rfe)
LR = sm.OLS(y_train,X_train_lm).fit()  
print(LR.summary())
array = df7.values
X = array[:,0:-1]
Y = array[:,-1]
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, random_state=51)
import warnings
warnings.filterwarnings('ignore')
result = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=51)    
    croresult = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')    
    result.append(croresult)
    output = "%s: %f (%f)" % (name, croresult.mean(), croresult.std())
    print(output)
LR = LogisticRegression()
LR.fit(X_train, Y_train)
predictions = LR.predict(X_test)
print(accuracy_score(Y_test, predictions))
from sklearn.metrics import confusion_matrix
import pylab as pl
cm = confusion_matrix(Y_test, predictions)
pl.matshow(cm)
pl.title('Confusion matrix \n')
pl.colorbar()
pl.show()
array = df7.values
X = array[:,0:-1]
Y = array[:,-1]
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, random_state=51)
from sklearn.ensemble import RandomForestClassifier
models = []
models.append(('RFC', RandomForestClassifier()))
result = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=51)
    croresults = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_mean_squared_error')
    result.append(croresults)
    output = "%s: %f (%f)" % (name, croresults.mean(), croresults.std())
    print(output)
RFC = RandomForestClassifier(n_estimators=50)
RFC.fit(X_train, Y_train)
predictions = RFC.predict(X_test)
print(accuracy_score(Y_test, predictions))
from sklearn.metrics import confusion_matrix
import pylab as pl
cm = confusion_matrix(Y_test, predictions)
pl.matshow(cm)
pl.title('Confusion matrix \n')
pl.colorbar()
pl.show()