import pandas as pd;

import matplotlib.pyplot as plt

import matplotlib



matplotlib.style.use('ggplot')
data = pd.read_csv("../input/HR_comma_sep.csv");

data.head()
data1 = data[['left', 'sales']]

data1 = data1.groupby('sales').agg(['sum', 'count']).reset_index()

data1.columns = data1.columns.get_level_values(0)

data1.columns = ['dep', 'left', 'total']

data1.plot(x='dep',y=['left', 'total'],kind='bar')

plt.show()
#Time spend at the company vs left! After 3 years at the company chances of leaving it decreases

data1 = data[['left', 'time_spend_company']]

data1 = data1.groupby('time_spend_company').agg(['sum', 'count']).reset_index()

data1.columns = data1.columns.get_level_values(0)

data1.columns = ['time_spend_company', 'left', 'total']

data1.plot(x='time_spend_company',y=['left'])

plt.show()
#compute correlation of left colum with other variables

corr = data.corr()

corr['left'].plot(kind='bar')

plt.show()
#Satisfication level vs time spent in the company!

data1 = data[['time_spend_company', 'satisfaction_level']]

data1 = data1.groupby('time_spend_company').agg(['mean']).reset_index()

data1.columns = data1.columns.get_level_values(0)

data1.plot(x='time_spend_company', y='satisfaction_level')

plt.show()
#Relation between number of employees leave the company in a relation with both their salaries

#and number of years in the company

data1 = data[['time_spend_company', 'salary','left']]

data1 = data1.groupby(['time_spend_company','salary']).agg(['sum']).reset_index()

data1.columns = data1.columns.get_level_values(0)

#convert salary column to numeric values high=>3, medium=>2, low=>1

data1.loc[:,'salary'] = data1.apply(lambda row : {'high':3, 'medium':2, 'low':1}[row['salary']], 1)

data1.plot(x='time_spend_company', y='salary', s=data1['left'], kind='scatter')

plt.show()
#Building first simple classifier 

from sklearn.naive_bayes import GaussianNB



#convert salary into numerica levels

data.loc[:,'salary'] = data.apply(lambda row : {'high':3, 'medium':2, 'low':1}[row['salary']], 1)

#drop department column

data = data.drop('sales', 1)

gnb = GaussianNB()



train = data.sample(frac=0.8, random_state=1)

test  = data.loc[~data.index.isin(train.index)]



#prediction based on: last_evaluation, number_project, average_montly_hours, 

#time_spend_company, salary

train_x = train[[1, 2, 3, 4, 8]]

test_x = test[[1, 2, 3, 4, 8]]



model = gnb.fit(train_x, train.left)

y_predict = model.predict(test_x)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



#print accuracy

accuracy_score(test.left, y_predict)
#Trying decision trees model

from sklearn import tree



dtc = tree.DecisionTreeClassifier()

dtm = dtc.fit(train_x, train.left)

y_predict = dtm.predict(test_x)



accuracy_score(test.left, y_predict)
#One Decision Trees disadvantage is that it can create over complix model that doesn't generalize

#well and cause the problem of overfitting.



#We will use here GridSearchCV method from sklearn to examin the whole space of two 

#params of descision trees to find the best compination of params

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report



max_depth      = range(1, 10)

max_leaf_nodes = range(2, 10) 



#we will tune only max_depth and max_leaf_nodes parameters of the DT. In the previous

#section we created a DecisionTreeClassifier without setting any of these parameters

#By default the classifier would use the maximum depth of trees to fit the data which

#would cause overfitting problem!

parameters = {'max_depth': max_depth, 'max_leaf_nodes': max_leaf_nodes}



gs = GridSearchCV(tree.DecisionTreeClassifier(), parameters, cv=5)

gs.fit(train_x, train.left)



#Best params

gs.best_params_
#Printing scores report of the best dt model found by the gridsearch

y_predict = gs.predict(test_x)

print(classification_report(test.left, y_predict))