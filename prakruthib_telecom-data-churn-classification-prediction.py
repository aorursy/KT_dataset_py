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
df= pd.read_csv("/kaggle/input/churn/churn.csv")
df.head()
df.isnull().sum()
df.duplicated().sum()
df.shape
df['international_plan'].value_counts()
df.describe()
df.dtypes
df['area_code'].value_counts()
df['state'].unique()
import matplotlib.pyplot as plt
#Exploratory data analysis
import seaborn as sns

sns.countplot(df['state'])
df['state'].value_counts()
sns.countplot(x='international_plan',data=df)
plt.hist(x='total_day_calls',data=df)
sns.countplot(x='international_plan',hue='churn',data=df)
pd.crosstab(df['international_plan'],df['churn'],normalize='index').round(4)*100
sns.countplot(x='number_customer_service_calls',hue='churn',data=df)
pd.crosstab(df['number_customer_service_calls'],df['churn'],normalize='index').round(4)*100
sns.countplot(x=df['voice_mail_plan'],hue='churn',data=df)
pd.crosstab(df['voice_mail_plan'],df['churn'],normalize='index').round(4)*100
pd.crosstab(df['state'],df['churn'],normalize='index').round(4)*100
##Insights

#customers with service calls 4 and more than 4 are tending to churn

#customers with voicecalls plan tend to churn less frequently

#customers with international plan are tending to churn more frequently
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

lb=LabelEncoder()

df['international_plan']=lb.fit_transform(df['international_plan'])

df['voice_mail_plan']=lb.fit_transform(df['voice_mail_plan'])

df['churn']=lb.fit_transform(df['churn'])
df.drop(columns=['state','account_length','area_code'],inplace=True,axis=1)
col = list(df.columns)

predictors = col[0:16]

target=col[16]
df.head()
df.shape
df.head()
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(new_df,df[target])
#standardizing the data



from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

d=sc.fit_transform(df[predictors])

new_df=pd.DataFrame(d)
lr = LogisticRegression()

model1 = lr.fit(xtrain,ytrain)

pred1 = model1.predict(xtest)
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score,accuracy_score,roc_auc_score,roc_curve

print(confusion_matrix(ytest,pred1))

print(f1_score(ytest,pred1))

print(precision_score(ytest,pred1))

print(recall_score(ytest,pred1))

print(accuracy_score(ytest,pred1))
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()

model2=knn.fit(xtrain,ytrain)

pred2=model2.predict(xtest)
print(confusion_matrix(ytest,pred2))

print(f1_score(ytest,pred2))

print(precision_score(ytest,pred2))

print(recall_score(ytest,pred2))

print(accuracy_score(ytest,pred2))
from sklearn.tree import DecisionTreeClassifier

dc = DecisionTreeClassifier(criterion='entropy')

model3= dc.fit(xtrain,ytrain)

pred3 = model3.predict(xtest)

print(confusion_matrix(ytest,pred3))

print(f1_score(ytest,pred3))

print(precision_score(ytest,pred3))

print(recall_score(ytest,pred3))

print(accuracy_score(ytest,pred3))
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
gb = GaussianNB()

model4= gb.fit(xtrain,ytrain)

pred4 = model4.predict(xtest)



bnb = BernoulliNB()

model5= bnb.fit(xtrain,ytrain)

pred5 = model5.predict(xtest)

#gaussian naive bayes

print(confusion_matrix(ytest,pred4))

print(f1_score(ytest,pred4))

print(precision_score(ytest,pred4))

print(recall_score(ytest,pred4))

print(accuracy_score(ytest,pred4))



#Bernaulli naive bayes

print(confusion_matrix(ytest,pred5))

print(f1_score(ytest,pred5))

print(precision_score(ytest,pred5))

print(recall_score(ytest,pred5))

print(accuracy_score(ytest,pred5))

from sklearn.ensemble import RandomForestClassifier
#random forest classifier

rf = RandomForestClassifier(n_estimators=100)

model6 = rf.fit(xtrain,ytrain)

pred6 = model6.predict(xtest)

print(confusion_matrix(ytest,pred6))

print(f1_score(ytest,pred6))

print(precision_score(ytest,pred6))

print(recall_score(ytest,pred6))

print(accuracy_score(ytest,pred6))



#ROC curve

ns_probs = [0 for _ in range(len(ytest))]

# predict probabilities

lr_probs = model6.predict_proba(xtest)

# keep probabilities for the positive outcome only

lr_probs = lr_probs[:, 1]

# calculate scores

ns_auc = roc_auc_score(ytest, ns_probs)

lr_auc = roc_auc_score(ytest, lr_probs)

# summarize scores

print('No Skill: ROC AUC=%.3f' % (ns_auc))

print('Random Forest: ROC AUC=%.3f' % (lr_auc))

# calculate roc curves

ns_fpr, ns_tpr, _ = roc_curve(ytest, ns_probs)

lr_fpr, lr_tpr, _ = roc_curve(ytest, lr_probs)

# plot the roc curve for the model

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')

plt.plot(lr_fpr, lr_tpr, marker='.', label='Random forest')

# axis labels

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

# show the legend

plt.legend()

# show the plot

plt.show()