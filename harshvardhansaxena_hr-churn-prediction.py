import pandas as pd 
import numpy as np
df = pd.read_csv('HR_Data.csv')
df
print(df.isnull().sum())
df.describe()
df.info()
df.groupby('Status').count()
df7 = df['Status']
df7 = pd.get_dummies(df7,drop_first=True)
df1 = pd.merge(df, df7, left_index = True, right_index = True)
#checking duplicates
sum(df1.duplicated(subset = 'SLNO'))
# having duplicate values
df1.drop_duplicates(subset ="SLNO",keep = False, inplace = True) 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import seaborn as sns
sns.distplot(df1['Age'])
sns.catplot("Not Joined",'Age',data = df1)
hike = plt.figure(figsize = (8,6))

g1 = hike.add_subplot(2,2,1) 
g2 = hike.add_subplot(2,2,2)

g1.hist(df1['Pecent.hike.expected.in.CTC'], color = 'green')
g1.set_title('Pecent.hike.expected.in.CTC')

g2.hist(df1['Percent.hike.offered.in.CTC'], color = 'red')
g2.set_title('Percent.hike.offered.in.CTC')

plt.tight_layout() 
plt.show()
scatter_age_balance = df1.plot.scatter('Pecent.hike.expected.in.CTC','Percent.hike.offered.in.CTC',color = 'blue',figsize = (7,5))

plt.title('The Relationship between offered and expected ')
plt.show()
dur_cam = sns.lmplot(x='Duration.to.accept.offer', y='Notice.period',data = df1,hue = 'Status',fit_reg = False,scatter_kws={'alpha':0.6},palette = 'rainbow', height =6)

plt.axis([0,65,0,65])
plt.ylabel('Notice.period')
plt.xlabel('Duration.to.accept.offer')
plt.title('The Relationship between the Number and Duration of Calls (with Response Result)')
plt.show()
sns.catplot("Not Joined",'Percent.difference.CTC',data = df1)
g= sns.pairplot(df1)
plt.figure(figsize=(30,30))
ax = sns.heatmap(df1.corr(), annot = True, linewidth = 3)
ax.tick_params(size = 10, labelsize = 10)
plt.title("HR CHURN", fontsize = 25)
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
import warnings
warnings.filterwarnings('ignore')
df2 = df1.copy()
df2
df2.drop(['Candidate.Ref'],axis=1, inplace=True)
df2
df2.drop(['Location'],axis=1, inplace=True)
df2.drop(['Status'],axis = 1, inplace = True)
df2 = pd.get_dummies(df2,drop_first=True)
df7= df2['Not Joined'].copy()
df7
df2.drop(['Not Joined'],axis = 1, inplace = True)
df2 = pd.merge(df2, df7, left_index = True, right_index = True)
df2.drop(['SLNO'],axis = 1, inplace = True)
df2
df2.info()
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
num_vars = ['Duration.to.accept.offer', 'Notice.period', 'Pecent.hike.expected.in.CTC', 'Percent.hike.offered.in.CTC', 'Percent.difference.CTC','Rex.in.Yrs', 'Age']

df2[num_vars] = sc.fit_transform(df2[num_vars])
list(zip(df2.columns))
col = df2.columns
col
X_sca = df2[col]
import statsmodels.api as sm  
X_sca = sm.add_constant(X_sca)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_sca
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
df2.drop(['LOB_MMS'],axis = 1, inplace = True)
df2.drop(['Gender_Male'],axis = 1, inplace = True)
df2.drop(['Joining.Bonus_Yes'],axis = 1, inplace = True)
df2.drop(['Candidate.relocate.actual_Yes'],axis = 1, inplace = True)
df2
array = df2.values
X = array[:,0:-1]
Y = array[:,-1]
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, random_state=51)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RFC', RandomForestClassifier()))
result = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=51)
    croresults = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    result.append(croresults)
    output = "%s: %f (%f)" % (name, croresults.mean(), croresults.std())
    print(output)
RFC = RandomForestClassifier(n_estimators=42)
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