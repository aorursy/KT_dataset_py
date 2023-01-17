#basic library 

import pandas as pd

import numpy as np





#visualization 

import matplotlib.pyplot as plt 

import seaborn as sns 

import cufflinks as cf 

from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot #iplot for interactive graphs

init_notebook_mode(connected=True)



cf.go_offline()

%matplotlib inline

sns.set_style('whitegrid')
df=pd.read_csv('../input/loan-data/loan_data.csv')

df.head(5)
df.describe().head(4)
NAN_value = (df.isnull().sum() / len(df)) * 100

Missing = NAN_value[NAN_value==0].index.sort_values(ascending=False)

Missing_data = pd.DataFrame({'Missing Ratio' :NAN_value})

Missing_data.head(20)
f,ax = plt.subplots(figsize = (15,10))

sns.heatmap(df.corr(),cmap='viridis',annot=True, ax=ax)

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)

print("Correlacion between variables")
f,(ax1,ax2,ax3)= plt.subplots(1,3,figsize=(25,10))

sns.distplot(df['int.rate'], bins= 30,ax=ax1)

sns.boxplot(data =df, x ='credit.policy', y= df['int.rate'],ax=ax2).legend().set_visible(False)

sns.boxplot(data = df['int.rate'], ax=ax3)

print("Interest Rate Distribution, Credit Policy range based on the Credit policy , General Interest rate")
#Finding Relationship between fisco ~ interest rate ~ installment 

sns.jointplot(y='int.rate', x='fico',data= df)

print("Interest rate - FICO ")
f,(ax4,ax5,ax6)= plt.subplots(1,3,figsize=(30,10))



sns.countplot(x='purpose',data=df, 

              hue='not.fully.paid',palette='Set1',ax=ax4)



sns.boxplot(x='purpose', y='int.rate',

            data= df, hue='not.fully.paid',palette='Set2',ax=ax5).legend().set_visible(False)



sns.boxplot(x='purpose', y='int.rate',

            data= df,ax=ax6)





print('Data1 : Reason of the Loan.    Data2: Interest based on the Reason')
sns.lmplot(data=df,palette='Set1',x='fico',y='int.rate', hue='credit.policy',col='not.fully.paid')

# Load the example mpg dataset

mpg = sns.load_dataset("mpg")



# Plot miles per gallon against horsepower with other semantics

sns.relplot(x="fico", y="int.rate",hue= 'not.fully.paid', sizes=(30, 200), alpha=.4, size ='installment',

         height=6, data=df )
plt.figure(figsize=(11,6))

fico_0 = df[df['not.fully.paid']==0]['fico'].hist(alpha=0.4,color='red',bins=30,label='not.fully.paid=1')

fico_1 = df[df['not.fully.paid']==1]['fico'].hist(alpha=0.4,color='blue',bins=30,label='not.fully.paid=0')

plt.legend()

plt.xlabel('FICO')
plt.figure(figsize=(11,6))

fico_0 = df[df['not.fully.paid']==0]['fico'].hist(alpha=0.4,color='red',bins=30,label='not.fully.paid=1')

fico_1 = df[df['not.fully.paid']==1]['fico'].hist(alpha=0.4,color='blue',bins=30,label='not.fully.paid=0')

plt.legend()

plt.xlabel('FICO')
# Attention somo outliers

out_1 = df[(df['fico']>750) & (df['int.rate']>0.175)].index.to_list()

out_2 = df[(df['fico']<700) & (df['int.rate']<0.075)].index.to_list()

Outliers = out_1 + out_2

df.iloc[Outliers]

loan_1 = df.drop(Outliers)
#Purpose to Categorical Data

final_loan  = pd.get_dummies(loan_1,drop_first = True) # without outliers

final_data = pd.get_dummies(df,drop_first = True)



# Applying Machine Learning 

from sklearn.model_selection import train_test_split

X=final_data.drop('not.fully.paid', axis=1)

y= final_data['not.fully.paid']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

y_pred = dtree.predict(X_test)



# Measuring Accurancy 

from sklearn.metrics import classification_report,confusion_matrix

dtree_score=classification_report(y_test,y_pred)

print(classification_report(y_test,y_pred))
X=final_loan.drop('not.fully.paid', axis=1)

y= final_loan['not.fully.paid']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



dtree_loan = DecisionTreeClassifier()

dtree_loan.fit(X_train,y_train)

y_pred_loan = dtree_loan.predict(X_test)

dtree_outlier = classification_report(y_test,y_pred_loan)

print(classification_report(y_test,y_pred_loan))
from sklearn.ensemble import RandomForestClassifier

X=final_data.drop('not.fully.paid', axis=1)

y= final_data['not.fully.paid']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

RFC = RandomForestClassifier(n_estimators = 50)

RFC.fit(X_train,y_train)

y_pred_RFC = (y_test)

RFC_report=classification_report(y_test,y_pred_RFC)

print(classification_report(y_test,y_pred_RFC))
from sklearn.ensemble import RandomForestClassifier

X=final_data.drop('not.fully.paid', axis=1)

y= final_data['not.fully.paid']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

RFC = RandomForestClassifier(n_estimators = 300)

RFC.fit(X_train,y_train)

y_pred_RFC = (y_test)

RFC_outliers = classification_report(y_test,y_pred_RFC)

print(classification_report(y_test,y_pred_RFC))
#Error_Rate = []

#for i in range (1,310):

    

 #   RFC_Error = RandomForestClassifier(n_estimators = i)

 #  RFC_Error.fit(X_train,y_train)

 #  pred_i = RFC_Error.predict(X_test)

 # Error_Rate.append(np.mean(pred_i != y_test))

    

#plt.figure(figsize=(10,6))

#plt.plot(range(1,310), Error_Rate , color = 'blue', linestyle = 'dashed', marker = 'o')