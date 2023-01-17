import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
dt=pd.read_csv('../input/bankfull/bank-full.csv')
dt.head()
dt['Target'].hist()
dt.isnull().sum()
dt.duplicated().sum()
fig = plt.figure(figsize = (15,20))

ax = fig.gca()

dt.hist(ax = ax)

plt.show()
plt.figure(figsize=(15,8))

sns.heatmap(dt.corr(),annot=True)

plt.show()
dt['default'] = dt['default'].map({'yes': 1, 'no': 0})

dt['housing'] = dt['housing'].map({'yes': 1, 'no': 0})

dt['loan'] = dt['loan'].map({'yes': 1, 'no': 0})

dt['Target'] = dt['Target'].map({'yes': 1, 'no': 0})
dt.drop(['day','month','marital'],1,inplace=True)
dt.shape
job = pd.get_dummies(dt['job'],prefix='job',drop_first=True)

dt = pd.concat([dt,job],axis=1)



edu = pd.get_dummies(dt['education'],prefix='education',drop_first=True)

dt = pd.concat([dt,edu],axis=1)



cont = pd.get_dummies(dt['contact'],prefix='contact',drop_first=True)

dt = pd.concat([dt,cont],axis=1)



pout = pd.get_dummies(dt['poutcome'],prefix='poutcome',drop_first=True)

dt = pd.concat([dt,pout],axis=1)
dt.drop(['job','education','contact','poutcome'], 1,inplace=True)
dt.info()
dt.head()
sns.boxplot(x=dt['age'])
sns.boxplot(x=dt['balance'])
sns.boxplot(x=dt['duration'])
sns.boxplot(x=dt['campaign'])
sns.boxplot(x=dt['pdays'])
sns.boxplot(x=dt['previous'])
from scipy import stats

z = np.abs(stats.zscore(dt[['age','balance','duration','campaign','pdays','previous']]))

print(z)
threshold = 3

print(np.where(z > 3))
dt.shape
df = dt[(z < 3).all(axis=1)]
df.shape
sns.boxplot(df['age'])
df.describe()
df.head()
plt.figure(figsize=(20,10))

sns.heatmap(df.corr(),annot=True)
X = df.drop(['Target'],1)

y = df['Target']
import statsmodels.api as sm



res = sm.Logit(y,X).fit()

res.summary()
X.drop(['poutcome_other','default'],1,inplace=True)
res = sm.Logit(y,X).fit()

res.summary()
X.drop(['education_unknown','contact_telephone'],1,inplace=True)
res = sm.Logit(y,X).fit()

res.summary()
X.drop(['previous','job_student'],1,inplace=True)
res = sm.Logit(y,X).fit()

res.summary()
from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler

from imblearn.pipeline import Pipeline

from collections import Counter
#The numbers before SMOTE

num_before = dict(Counter(y))



#Performing SMOTE



#Define pipeline

over = SMOTE(sampling_strategy=0.8)

under = RandomUnderSampler(sampling_strategy=0.8)

steps = [('o', over), ('u', under)]

pipeline = Pipeline(steps=steps)



#Transforming the dataset

X_smote, y_smote = pipeline.fit_resample(X, y)





#Tthe numbers after SMOTE

num_after =dict(Counter(y_smote))

print(num_before, num_after)
labels = ["No","Yes"]

plt.figure(figsize=(15,6))

plt.subplot(1,2,1)

sns.barplot(labels, list(num_before.values()))

plt.title("Numbers Before Balancing")

plt.subplot(1,2,2)

sns.barplot(labels, list(num_after.values()))

plt.title("Numbers After Balancing")

plt.show()
X1 = pd.DataFrame(X_smote)

y1= pd.DataFrame(y_smote)
new_data = pd.concat([X1, y1], axis=1)

new_data.columns = ['age', 'balance', 'housing', 'loan', 'duration', 'campaign', 'pdays','job_blue-collar','job_entrepreneur','job_housemaid',

                    'job_management','job_retired','job_self-employed','job_services','job_technician','job_unemployed','job_unknown','education_secondary',

                    'education_tertiary','contact_unknown','poutcome_success','poutcome_unknown','outcome']

new_data.head()
X_new = new_data.drop(['outcome'],1)

y_new = new_data['outcome']
X_new.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_new,y_new,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = 0)

lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

print(y_pred)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

cm = confusion_matrix(y_test, y_pred)

acc = accuracy_score(y_test, y_pred)

pre = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



print("Confusion Matrix : \n",cm)

print("Accuracy :",acc)

print("Precision :",pre)

print("Recall :",recall)

print("F1-score :",f1)