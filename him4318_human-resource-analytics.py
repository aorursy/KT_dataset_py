import pandas as pd

from sklearn.ensemble import RandomForestClassifier



import matplotlib.pyplot as plt

import matplotlib.pylab as plb

import matplotlib as mpl

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

import numpy as np

%matplotlib inline
data = pd.read_csv('../input/HR_comma_sep.csv')

data["salary"] = data["salary"].replace(['low' , 'medium' , 'high'] , [0 , 1 , 2])



train=data.sample(frac=0.8,random_state=200)

test = data.drop(train.index)

data.head()
traininput1 = train[['satisfaction_level', 'last_evaluation', 'number_project',

       'average_montly_hours', 'time_spend_company', 'Work_accident',

       'promotion_last_5years', 'salary']].values

traintarget = train[["left"]].values

testinput1 = test[['satisfaction_level', 'last_evaluation', 'number_project',

       'average_montly_hours', 'time_spend_company', 'Work_accident',

       'promotion_last_5years', 'salary']].values

testtarget = test[["left"]].values



forest = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators = 100, random_state = 1)

my_forest = forest.fit(traininput1,traintarget)

print("Testing accuracy :" , my_forest.score(testinput1 , testtarget))

print("Training accuracy :" , my_forest.score(traininput1, traintarget))



features = list(my_forest.feature_importances_)

print("Importance of feattures:")

print("'satisfaction_level :{f[0]}', 'last_evaluation : {f[1]}', 'number_project: {f[2]}','average_montly_hours: {f[3]}', 'time_spend_company: {f[4]}', 'Work_accident: {f[5]}', 'promotion_last_5years: {f[6]}', 'salary: {f[7]}'".format(f = features))





traininput2= train[['satisfaction_level', 'last_evaluation', 'number_project',

       'average_montly_hours', 'time_spend_company'

       ]].values

testinput2= test[['satisfaction_level', 'last_evaluation', 'number_project',

       'average_montly_hours', 'time_spend_company'

       ]].values



forest2 = RandomForestClassifier( max_depth=10 , min_samples_split=2, n_estimators = 100, random_state = 1)



my_forest2 = forest2.fit(traininput2,traintarget)

print("Testing accuracy :" , my_forest2.score(testinput2 , testtarget))

print("Training accuracy :" , my_forest2.score(traininput2, traintarget))



features1 =  list(my_forest2.feature_importances_)

print("How important a feature is :")

print("'satisfaction_level :{f[0]}', 'last_evaluation : {f[1]}', 'number_project: {f[2]}','average_montly_hours: {f[3]}','time_spend_company: {f[4]}',".format(f = features1))

plt.figure(figsize = (18,8))

plt.suptitle('Employees who left', fontsize=16)

plt.subplot(1,4,1)

plt.plot(data.satisfaction_level[data.left == 1],data.last_evaluation[data.left == 1],'o', alpha = 0.1)

plt.ylabel('Last Evaluation')

plt.title('Evaluation vs Satisfaction')

plt.xlabel('Satisfaction level')



plt.subplot(1,4,2)

plt.plot(data.satisfaction_level[data.left == 1],data.average_montly_hours[data.left == 1],'o', alpha = 0.1 )

plt.ylabel('Average Monthly Hours')

plt.title('Average hours vs Satisfaction ')

plt.xlabel('Satisfaction level')



plt.subplot(1,4,3)

plt.title('Salary vs Satisfaction ')

plt.plot(data.satisfaction_level[data.left == 1],data.salary[data.left == 1],'o', alpha = 0.1)

plt.xlim([0.4,1])

plt.ylabel('salary ')

plt.xlabel('Satisfaction level')



plt.subplot(1,4,4)

plt.title('Promotions vs Satisfaction ')

plt.plot(data.satisfaction_level[data.left == 1],data.promotion_last_5years[data.left == 1],'o', alpha = 0.1)

plt.xlim([0.4,1])

plt.ylabel('Promotion last 5years')

plt.xlabel('Satisfaction level')





import seaborn as sns



correlation = data.corr()

plt.figure(figsize=(12,12))

sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='terrain')

plt.title('Correlation between different fearures' )
from sklearn import linear_model

logreg = linear_model.LogisticRegression( C=1000)

logreg.fit(traininput2, traintarget)

probability = logreg.predict_proba(testinput2)

new =pd.DataFrame(list(probability) ,columns=['Stayed','left'])



plt.figure(figsize=(20,10))

plt.subplot(1,2,1)

plt.title("Employee who left vs Average hours ")

plt.xlabel("Probability of leaving")

plt.ylabel("Average hours")

plt.plot(new["left"] , test["average_montly_hours"] , 'o' , alpha= .3 )



plt.subplot(1,2,2)

plt.title("Employee who left vs Satisfaction level ")

plt.xlabel("Probability of leaving")

plt.ylabel("Satisfiction level")

plt.plot(new["left"] , test["satisfaction_level"] , 'o' , alpha= .3)


