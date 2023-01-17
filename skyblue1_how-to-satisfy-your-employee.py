import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/HR_comma_sep.csv')

df.describe()
df.isnull().any()


#The meanings of all the columns are quite understandable except 'sales' which actually represents 

#the department of the employee

df=df.rename(columns={'sales':'job'})
sns.heatmap(df.corr(), vmax=.8, square=True,annot=True,fmt='.2f')
#satisfaction level comes out as the most correlated feature



from sklearn.ensemble import ExtraTreesClassifier

from sklearn.preprocessing import LabelEncoder 





le=LabelEncoder()

df['job']= le.fit_transform(df['job'])

df['salary']= le.fit_transform(df['salary'])
X= np.array(df.drop('left',1))

y=np.array(df['left'])



model= ExtraTreesClassifier()

model.fit(X,y)



feature_list= list(df.drop('left',1).columns)



feature_importance_dict= dict(zip(feature_list,model.feature_importances_))



print(sorted(feature_importance_dict.items(), key=lambda x: x[1],reverse=True))
sns.barplot(df['left'],df['satisfaction_level'])

#More the satisfaction level lesser the chance of an employee to leave.
facet = sns.FacetGrid(df, hue="left",aspect=3)

facet.map(sns.kdeplot,'satisfaction_level',shade= True)

facet.set(xlim=(0, 1))

facet.add_legend()



#3 peaks for left=1 indicates 3 types of trends for an employee to leave.
from sklearn.ensemble import ExtraTreesRegressor



model= ExtraTreesRegressor()



X=df.drop(['left','satisfaction_level'],axis=1)

y=df['satisfaction_level']

model.fit(X,y)



feature_list= list(df.drop(['left','satisfaction_level'],1).columns)



feature_importance_dict= dict(zip(feature_list,model.feature_importances_))



print(sorted(feature_importance_dict.items(), key=lambda x: x[1],reverse=True))
#sns.swarmplot(x=[df['average_montly_hours'],df['last_evaluation']], y=df['satisfaction_level'])

plt.scatter(df['satisfaction_level'],df['average_montly_hours'])

plt.ylabel('average_montly_hours')

plt.xlabel('satisfaction_level')

plt.scatter(df['satisfaction_level'],df['last_evaluation'])

plt.xlabel('satisfaction_level')

plt.ylabel('last_evaluation')

sns.pointplot(df['number_project'],df['satisfaction_level'])

projects=df['number_project'].unique()

projects=sorted(projects)

for i in projects:

    mean_satisfaction_level=df['satisfaction_level'][df['number_project']==i].mean()

    print('project_total',i,':',mean_satisfaction_level)



#Expected reuslt
df1=df.copy()



group_name=list(range(20))

df1['last_evaluation']=pd.cut(df1['last_evaluation'],20,labels=group_name)

df1['average_montly_hours']=pd.cut(df1['average_montly_hours'],20,labels=group_name)



#average_monthly_hours bins:

"""

{0: '(149.5, 160.2]', 1: '(256.5, 267.2]', 2: '(267.2, 277.9]', 3: '(213.7, 224.4]', 4: '(245.8, 256.5]', 5: '(138.8, 149.5]',

 6: '(128.1, 138.8]', 7: '(299.3, 310]', 8: '(224.4, 235.1]', 9: '(277.9, 288.6]', 10: '(235.1, 245.8]'

 , 11: '(117.4, 128.1]', 12: '(288.6, 299.3]', 13: '(181.6, 192.3]', 14: '(160.2, 170.9]',

 15: '(170.9, 181.6]', 16: '(192.3, 203]', 17: '(203, 213.7]', 18: '(106.7, 117.4]',

 19: '(95.786, 106.7]'}

 """
sns.pointplot(df1['last_evaluation'],df1['satisfaction_level'])
#3 types of employees: 

#last_evaluation(0-3): satisfaction level is pretty low--> possibly not able to perform well 

#last_evaluation(7-12): satisfaction level is high---> possibly getting appreciated for their work 

#last_evaluation(13-18):satisfaction level is low again---> Not getting enough appreciated for their work**

sns.pointplot(df1['average_montly_hours'],df1['satisfaction_level'])
sns.pointplot(df['number_project'],df['last_evaluation'])

#As the number of projects increase last_evaluation score also increases.
sns.pointplot(df1['last_evaluation'],df['average_montly_hours'])



#more the hours you work higher is your last_evaluation score.
#Let's check some other features also.
sns.barplot(df['Work_accident'],df['satisfaction_level'])
sns.barplot(df['salary'],df['satisfaction_level'])
sns.barplot(df['job'],df['satisfaction_level'])
from sklearn.model_selection import KFold,cross_val_score,train_test_split

from sklearn.metrics import classification_report

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor





X=df.drop(['left','satisfaction_level'],axis=1)

y=df['satisfaction_level']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=7)



kfold=KFold(n_splits=10,random_state=7)

models=[['LR',LinearRegression()],['CART',DecisionTreeRegressor()],['RF',RandomForestRegressor()]]

scoring='neg_mean_squared_error'

result_list=[]

for names,model in models:

    results= cross_val_score(model, X,y, cv=kfold,scoring=scoring)

    print(names,results.mean())



    

#RandomForest performs the best. 

    

    
#Let's take a small example



test_dict={'last_evaluation':[0.2,0.6,0.7,0.8],'number_project':[1,3,4,6],'average_montly_hours':[110,180,190,250],

           'time_spend_company':[3,4,5,6],'Work_accident':[0,1,1,0],'promotion_last_5years':[0,0,1,1],'job':[0,1,2,3],

           'salary':[0,1,1,0]}



#1st employee is the stuggling one.

#2nd and 3rd fullfill all the required criterias for satisfaction level to be high.

#4th employee is a high performer.



df_test= pd.DataFrame(test_dict)



test_X= np.array(df_test)



model= RandomForestRegressor()

model.fit(X_train,y_train)

model.predict(test_X)