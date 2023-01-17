import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import export_graphviz

import seaborn as sns

import matplotlib.pyplot as plt
df= pd.read_csv('../input/HR_comma_sep.csv')

df.head(10)
df.isnull().sum() # Are there any empty variables?
df.dtypes # What are the data types?
df['sales'].unique() # Check the unique values in the 'Sales' column.
df['salary'].unique() # Check the unique values in the 'Salary' column.
df['salary'].replace({'low':-1,'medium':0,'high':1},inplace=True) # replace the variables (Low=-1, Medium=0, High=1)
satisfaction_level=df['satisfaction_level']

last_evaluation=df['last_evaluation']

number_project=df['number_project']

average_montly_hours=df['average_montly_hours']

time_spend_company=df['time_spend_company']

Work_accident=df['Work_accident']
sns.pairplot(df, hue="left", palette="husl", vars=['satisfaction_level', 'last_evaluation', 'average_montly_hours','number_project'])

plt.show()



# Make plots
#X_train,y_train,X_test,y_test=train_test_split(X,y,data_size=0.2)



corr=df.corr()

sns.heatmap(corr,annot=True) #test correlations
sns.pairplot(df, hue="salary", vars=['satisfaction_level','time_spend_company', 'last_evaluation', 'average_montly_hours','number_project'])

plt.show()
#Creating dummy variables for the column:sales

dummies=pd.get_dummies(df['sales'],prefix='sales')

df=pd.concat([df,dummies],axis=1)

df.drop(['sales'],axis=1,inplace=True)

df.head(10)
#Spilting data into test and train split:

X=df.drop(['left'],axis=1)

y=df['left']
#RandomForestClassifier

model=RandomForestRegressor(n_estimators=100,n_jobs=-1,oob_score=True,random_state=19)



model.fit(X,y)
#RandomForestClassifier

from sklearn.metrics import roc_auc_score





y_pred=model.oob_prediction_



acc=roc_auc_score(y,y_pred)

print('Accuracy :',acc)
#K-Fold Validation

#Evaluating the model for a Ten-Fold Cross Validation



acc=[]

xx=[50,100,150,200,250,300,350,400,500,600]

for i in xx:

    #Training the model

    clf=RandomForestRegressor(n_estimators=i,n_jobs=-1,oob_score=True,random_state=19)

    clf.fit(X,y)

    y_pred=clf.oob_prediction_

    k=roc_auc_score(y,y_pred)

    acc.append(k)

   

print(acc)

    

    

    
import matplotlib.pyplot as plt

plt.plot(xx,acc)