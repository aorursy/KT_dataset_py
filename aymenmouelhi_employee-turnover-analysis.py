import numpy as np #linear algebra

import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)



#data visualization

import seaborn as sns 

import matplotlib.pyplot as plt



#machine learning techniques

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
df=pd.read_csv('../input/HR_comma_sep.csv')

df.head()
df=df.rename(columns={'average_montly_hours':'average_weekly_hours','sales':'department'})

df['average_weekly_hours']=df['average_weekly_hours']*12/52

df.info()
print (np.corrcoef(df['number_project'], df['average_weekly_hours']))
df.describe()
df.describe(include=['O'])
df[['Work_accident', 'left']].groupby(['Work_accident'], as_index=False).mean().sort_values(by='left')
df[['department', 'left']].groupby(['department'], as_index=False).mean().sort_values(by='left', ascending=False)
df[['salary', 'left']].groupby(['salary'], as_index=False).mean().sort_values(by='left', ascending=False)
df[['number_project', 'left']].groupby(['number_project'], as_index=False).mean().sort_values(by='number_project')
df[['time_spend_company', 'left']].groupby(['time_spend_company'], as_index=False).mean().sort_values(by='time_spend_company')
leave_sat=df.groupby('satisfaction_level').agg({'left': lambda x: len(x[x==1])})

leave_sat['total']=df.groupby('satisfaction_level').agg({'left': len})

leave_sat['leave_rate']=leave_sat['left']/leave_sat['total']

leave_sat['satisfaction']=df.groupby('satisfaction_level').agg({'satisfaction_level': 'mean'})

g=sns.lmplot('satisfaction', 'leave_rate',data=leave_sat)
leave_eval=df.groupby('last_evaluation').agg({'left': lambda x: len(x[x==1])})

leave_eval['total']=df.groupby('last_evaluation').agg({'left': len})

leave_eval['leave_rate']=leave_eval['left']/leave_eval['total']

leave_eval['evaluation']=df.groupby('last_evaluation').agg({'last_evaluation': 'mean'})

gr=sns.lmplot('evaluation', 'leave_rate',data=leave_eval,fit_reg=False)
leave_hours=df.groupby('average_weekly_hours').agg({'left': lambda x: len(x[x==1])})

leave_hours['total']=df.groupby('average_weekly_hours').agg({'left': len})

leave_hours['leave_rate']=leave_hours['left']/leave_hours['total']

leave_hours['weekly_hours']=df.groupby('average_weekly_hours').agg({'average_weekly_hours': 'mean'})

grid=sns.lmplot('weekly_hours', 'leave_rate',data=leave_hours,fit_reg=False)
df[['department', 'average_weekly_hours']].groupby(['department'], as_index=False).mean().sort_values(by='average_weekly_hours', ascending=False)
(df.promotion_last_5years==1).sum()

df=df.drop(['promotion_last_5years'],axis=1)
df=df.drop(['Work_accident','department','average_weekly_hours'],axis=1)

df.columns
#banding number of projects

bins=[0,2,5,10]

names=[1,0,1]

df['abnormal_proj']=pd.cut(df['number_project'],bins,labels=names)

#banding years at the firm

bins2=[0,1,2,3,4,5,6,100]

names2=['1','2','3','4','5','6','7']

df['years_at_company']=pd.cut(df['time_spend_company'],bins2,labels=names2)

#banding last_evaluation

bins3=[0,.6,.8,1]

names3=[1,0,1]

df['abnormal_eval']=pd.cut(df['last_evaluation'],bins3,labels=names3)

df.head()
#cleaning up intermediary/unused columns

df=df.drop(['number_project','time_spend_company','last_evaluation'],axis=1)

df.head()
#turning all columns into numeric so that modeling algorithms can run

df['salary']=df['salary'].map({'low':0,'medium':1,'high':2}).astype(int)

pd.to_numeric(df['abnormal_proj'], errors='coerce')

pd.to_numeric(df['years_at_company'], errors='coerce')

pd.to_numeric(df['abnormal_eval'], errors='coerce')

df.head()
#Modeling

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(df,df['left'],test_size=.2)

X_train=X_train.drop('left',axis=1)

X_test=X_test.drop('left',axis=1)

print (X_train.shape, Y_train.shape)

print (X_test.shape, Y_test.shape)
#Log reg

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(X_train.columns)

coeff_df.columns = ['Feature']

coeff_df["Coefficient"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Coefficient', ascending=False)
#KNN

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
#SVM

svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
#NB

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
#Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Naive Bayes', 'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_gaussian,acc_decision_tree]})

models.sort_values(by='Score', ascending=False)