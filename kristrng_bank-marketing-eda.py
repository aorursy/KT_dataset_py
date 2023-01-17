import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
sns.set(style='whitegrid')
%matplotlib inline   
df = pd.read_csv('../input/bank-marketing-dataset/bank.csv')
df.head()
df.info()
#It seems that there is no null data. Just to be sure, let's check again
df.isnull().sum()
#Numerical exploration
df.describe()
# Check the mean of the numerical attributes above
df.mean()
df.hist(figsize=(14,10),bins=15,color='g')
plt.figure(figsize=(15,5))
sns.violinplot(x='job',y='balance',data=df,palette='Set2',)
plt.title('Distribution of balace by Job')
plt.figure(figsize=(8,4))
sns.violinplot(x='education',y='balance',data=df,palette='Set2',)
plt.title('Distribution of balace by education')
plt.figure(figsize=(10,6))
sns.scatterplot(x='age',y='balance',data=df,palette='Set2',hue='marital')
plt.title('Age vs Balance')
df.head()
#Plot the categorical attributes
plt.figure(figsize = (20,15))

plt.subplot(331)
df["job"].value_counts().plot.barh()
plt.title('Job Categories')

plt.subplot(332)
df["marital"].value_counts().plot.barh()
plt.title('Marital Status')

plt.subplot(333)
df["education"].value_counts().plot.barh()
plt.title('Education Levels')

plt.subplot(334)
df["default"].value_counts().plot.barh()
plt.title('Has Credit in Default')


plt.subplot(335)
df["housing"].value_counts().plot.barh()
plt.title('Has Housing Loan')

plt.subplot(336)
df["loan"].value_counts().plot.barh()
plt.title('Has Personal Loan')

plt.subplot(337)
df["contact"].value_counts().plot.barh()
plt.title('Contact Communication Type')

plt.subplot(338)
df["month"].value_counts().plot.barh()
plt.title('Months Value Counts')

plt.subplot(339)
df["poutcome"].value_counts().plot.barh()
plt.title('Outcome of Previous Marketing Campaign');

plt.plot()
#Check how many customers open Deposit
plt.figure(figsize=(8,4))
sns.countplot(x='deposit',data=df,palette='Set2')
plt.title('How many customers open the Term Deposit')
#Marital, education and contact, Default, housing and loan vs Y
plt.figure(figsize=[18,8])

plt.subplot(231)
sns.countplot(x='marital', hue='deposit', data=df,palette="Set2")

plt.subplot(232)
sns.countplot(x='education', hue='deposit', data=df,palette="Set2")

plt.subplot(233)
sns.countplot(x='contact', hue='deposit', data=df,palette="Set2")

plt.subplot(234)
sns.countplot(x='default', hue='deposit', data=df,palette="Set2")

plt.subplot(235)
sns.countplot(x='housing', hue='deposit', data=df,palette="Set2")

plt.subplot(236)
sns.countplot(x='loan', hue='deposit', data=df,palette="Set2")
#Job and Month vs Y
plt.figure(figsize=(14,12))

plt.subplot(211)
sns.countplot(y='job',data=df,hue='deposit',palette='Set2')
plt.title('Job vs Term Deposit')

plt.subplot(212)
sns.countplot(x='month',data=df,hue='deposit',palette='Set2')
plt.title('Last contact month vs Term Deposit')
#Last contact day vs Y
plt.figure(figsize=(17,5))
sns.countplot(x='day',data=df,hue='deposit',palette='Set2')
plt.title('Last contact day vs Term Deposit')
#Poutcome vs Y
plt.figure(figsize=(17,5))
sns.countplot(x='poutcome',data=df,hue='deposit',palette='Set2')
plt.title('Outcome of the previous campaign vs Term Deposit')
#Age against Y
g = sns.FacetGrid(data=df,hue='deposit',height=4,aspect=2)
g.map(sns.kdeplot,'age',shade=True,legend=True)
g.add_legend()
plt.title('Age against Y')
#Balance against Y
g = sns.FacetGrid(data=df,hue='deposit',height=4,aspect=2)
g.map(sns.kdeplot,'balance',shade=True,legend=True)
g.add_legend()
plt.title('Balance against Y')
#Number of contact performed for this campaign against Y
g = sns.FacetGrid(data=df,hue='deposit',height=4,aspect=2)
g.map(sns.kdeplot,'campaign',shade=True,legend=True)
g.add_legend()
plt.title('Number of contact performed during this campaign')
#Duration of the last contact against Y
g = sns.FacetGrid(data=df,hue='deposit',height=4,aspect=2)
g.map(sns.kdeplot,'duration',shade=True,legend=True)
g.add_legend()
plt.title('Duration of the last contact')
plt.plot()
sns.kdeplot(df[df['deposit']=='yes']['pdays'])
sns.kdeplot(df[df['deposit']=='no']['pdays'])
sns.distplot(df[df['deposit']=='no']['pdays']).plot()
#Pdays against Y
g = sns.FacetGrid(data=df,hue='deposit',height=4,aspect=2)
g.map(sns.kdeplot,'pdays',shade=True,legend=True)
g.add_legend()
plt.title('Number of days that passed by after the client was last contacted')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
#Replace "yes", "no" in deposit with 1,0
df2=df.copy()
df2.replace({'deposit': {"yes": 1,'no':0}},inplace=True)
df2
# Pre-processing data
df2 = pd.get_dummies(df2,drop_first=True)
df2
X = df2.drop(['deposit','duration'],axis=1) #As state in the guidance, we shouldn't use duration
y= df2['deposit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
print('Report:\n',classification_report(y_test, y_pred))
print('confusion Matrix:\n',confusion_matrix(y_pred,y_test))
#Top 5 important features

importances=rfc.feature_importances_
feature_importances=pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
sns.barplot(x=feature_importances[0:5], y=feature_importances.index[0:5])
plt.title('Feature Importance')
plt.ylabel("Features")