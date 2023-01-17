#import a few libraries that will be used

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns



#read in the data and examine the first 5 observations to get a feel for it

df = pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv") 

df.head()
#Assess how many null values are in the dataframe

df.isnull().sum()
df = df.fillna(0) #fill in null salary values with 0 since it's for students that were not placed.

df.describe()
fig = plt.figure()

fig.patch.set_facecolor('black')

plt.rcParams['text.color'] = 'white'

my_circle=plt.Circle( (0,0), 0.7, color='black')



plt.pie(df['gender'].value_counts(), labels=df['gender'].value_counts().index, autopct="%1.1f%%")

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()



fig = plt.figure()

fig.patch.set_facecolor('black')

plt.rcParams['text.color'] = 'white'

my_circle=plt.Circle( (0,0), 0.7, color='black')



plt.pie(df['status'].value_counts(), labels=df['status'].value_counts().index, autopct="%1.1f%%")

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()



fig = plt.figure()

fig.patch.set_facecolor('black')

plt.rcParams['text.color'] = 'white'

my_circle=plt.Circle( (0,0), 0.7, color='black')



plt.pie(df['specialisation'].value_counts(), labels=df['specialisation'].value_counts().index, autopct="%1.1f%%")

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()



fig = plt.figure()

fig.patch.set_facecolor('black')

plt.rcParams['text.color'] = 'white'

my_circle=plt.Circle( (0,0), 0.7, color='black')



plt.pie(df['workex'].value_counts(), labels=df['workex'].value_counts().index, autopct="%1.1f%%")

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()



fig = plt.figure()

fig.patch.set_facecolor('black')

plt.rcParams['text.color'] = 'white'

my_circle=plt.Circle( (0,0), 0.7, color='black')



plt.pie(df['degree_t'].value_counts(), labels=df['degree_t'].value_counts().index, autopct="%1.1f%%")

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()
sns.set(style="darkgrid")

ax = sns.countplot(x="gender", hue="status", data=df) #plot placement status by gender
sns.boxplot( x="salary", y='gender', width =0.5, data = df); #boxplot of salary by gender

plt.show()
df1 = df

df1['gender'].replace({"M":0, "F":1}, inplace=True)

df1['hsc_b'].replace({"Others":0, "Central":1}, inplace=True)

df1['workex'].replace({"No":0, "Yes":1}, inplace=True)

df1['status'].replace({"Not Placed":0, "Placed":1}, inplace=True)

df1['specialisation'].replace({"Mkt&HR":0, "Mkt&Fin":1}, inplace=True)



df1['hsc_s_commerce'] = np.where(df1['hsc_s'] == 'Commerce',1,0)

df1['hsc_s_Science'] = np.where(df1['hsc_s'] == 'Science',1,0)

df1['hsc_s_Arts'] = np.where(df1['hsc_s'] == 'Arts',1,0)

df1['degree_sci&tech'] = np.where(df1['degree_t'] == 'Sci&Tech',1,0)

df1['degree_comm&mgmt'] = np.where(df1['degree_t'] == 'Comm&Mgmt',1,0)

df1['degree_other'] = np.where(df1['degree_t'] == 'Others',1,0)



df1=df1.drop(['sl_no','ssc_p','ssc_b','hsc_s','degree_t','salary'],axis=1)
#import some sklearn modules in order to perform the regresison

from sklearn import metrics 

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

df1.head() #look at the first 5 observations to ensure coding is correct
x = df1.drop('status', axis=1) #create df with only effect variables

y = df1.status #create df with only response variable

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4) #split frames into testing and training

logistic_regression = LogisticRegression(max_iter=500)
logistic_regression.fit(x_train, y_train) #fit the training df's to a logistic regression model
y_pred = logistic_regression.predict(x_test) #predict y values with the model that was fit above

accuracy = metrics.accuracy_score(y_test, y_pred) #calculate accuracy of the prediction to actual values

accuracy_percentage = 100 * accuracy

print(accuracy_percentage)
import statsmodels.api as sm

logit_model=sm.Logit(y_train,x_train)

result=logit_model.fit()

print(result.summary2())

print(logit_model.fit().params)