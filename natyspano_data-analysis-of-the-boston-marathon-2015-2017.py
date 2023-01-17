import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import colorsys
import seaborn as sns
import warnings
from datetime import datetime
%matplotlib inline
warnings.simplefilter("ignore")
Boston_15 = pd.read_csv('../input/marathon_results_2015.csv', sep=',')
Boston_16 = pd.read_csv('../input/marathon_results_2016.csv', sep=',')
Boston_17 = pd.read_csv('../input/marathon_results_2017.csv', sep=',')
# The total dataset:
Boston_15_to_17 = pd.concat([Boston_15, Boston_16, Boston_17], ignore_index=True, sort=False).set_index('Name')
Boston_15_to_17.head()
#Checking the existence of the null values in the dataset
Boston_15_to_17.isnull().sum(axis=0)
Boston_15_to_17.columns
Boston = Boston_15_to_17.drop(['Pace','Unnamed: 0','Bib', 'Gender','Unnamed: 9', 'Division', 'State', 'Citizen','Proj Time','City', 'Unnamed: 8','5K','15K', '10K', '25K', '20K', 'Half', '30K', '35K', '40K', 'Overall'], axis='columns')
Boston.head()
#Checking the existence of the null values in the dataset
Boston.isnull().sum(axis=0)
# Changing the str columns to time form 
Boston['Official Time'] = pd.to_timedelta(Boston['Official Time'])
# Transforming the time in minutes:
Boston['Official Time'] = Boston['Official Time'].astype('m8[m]').astype(np.int32)
Boston.info()
Boston.describe()
print('The oldest person finishing the Boston Marathon 2015-2017 was {} years old.\nThe youngest person was {} years old.'.format(Boston['Age'].max(), Boston['Age'].min()))
plt.figure(figsize=(8,6))
hage = sns.distplot(Boston.Age, color='g')
hage.set_xlabel('Ages',fontdict= {'size':14})
hage.set_ylabel(u'Distribution',fontdict= {'size':14})
hage.set_title(u'Distribution of finishers for Ages',fontsize=18)
plt.show()
warnings.simplefilter("ignore")
plt.figure(figsize=(20,10))
agecont = sns.countplot('Age',data=Boston, palette=sns.color_palette("RdPu", n_colors=len(Boston['Age'].value_counts())))
agecont.set_title('Ages Counting', fontsize=18)
agecont.set_xlabel('Ages', fontdict= {'size':16})
agecont.set_ylabel('Total of People', fontdict= {'size':16})
plt.show()
plt.figure(figsize=(25,25))
d = sns.countplot(x='Age', hue='M/F', data=Boston, palette={'F':'r','M':'b'}, saturation=0.6)
d.set_title('Number of Finishers for Age and Gender', fontsize=25)
d.set_xlabel('Ages',fontdict={'size':20})
d.set_ylabel('Number of Finishers',fontdict={'size':20})
d.legend(fontsize=16)
plt.show()
plt.figure(figsize=(6,6))
l = Boston['M/F'].value_counts().index
plt.pie(Boston['M/F'].value_counts(), colors =['b','r'], startangle = 90, autopct='%.2f', textprops=dict(color="w"))
#plt.axes().set_aspect('equal','datalim')
plt.legend(l, loc='upper right')
plt.title("Gender",fontsize=18)
plt.show()
Boston_1 = Boston.copy()
bins = [17, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 90]
Boston_1['Ranges'] = pd.cut(Boston_1['Age'],bins,labels=["18-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64",
                                          "65-69", "70-74", "75-79", "< 80"]) 

Boston_2 = pd.crosstab(Boston_1.Ranges,Boston_1['M/F']).apply(lambda r: (r/r.sum()) * 100 , axis=1)

ax1 = Boston_2.plot(kind = "bar", stacked = True, color = ['r','b'], figsize=(9,6),
                      fontsize=12, position=0.5)
ax1.get_legend_handles_labels
ax1.legend(bbox_to_anchor = (1.3, 1))
ax1.set_xlabel('Age Ranges', fontdict={'size':14})
ax1.set_ylabel('Percentages (%)', fontdict={'size':14})
ax1.set_title('Gender Finishers x Age Ranges', fontsize=18)
plt.show()
FM_mean = Boston.groupby('M/F').mean()
FM_mean
print('The average age of the female finishers in Boston is {:.2f} years old.'.format(FM_mean['Age'][0]))
print('The average age of the male finishers in Boston is {:.2f} years old.'.format(FM_mean['Age'][1]))
print('The average finishing time of the female finishers in Boston is {:.2f} hours.'.format(FM_mean['Official Time'][0] / 60))
print('The average finishing time of the male finishers in Boston is {:.2f} hours.'.format(FM_mean['Official Time'][1] / 60))
plt.figure(figsize=(12,10))
Boston_copy = Boston.copy()
Boston_copy = Boston_copy[Boston_copy['Age'].isin(range(0,85))]

x = Boston_copy.Age
y = Boston_copy['Official Time']


plt.plot(x, y, '.')
plt.xlabel("Age", fontsize=16)
plt.ylabel("Official Time (min)",fontsize=16)
plt.title("Official Time for Age",fontsize=20)
plt.show()
# The mean of official time for the set of Age 
mean_age_time = Boston.groupby('Age').mean().set_index(np.arange(67))
mean_age_time['Age'] = mean_age_time.index 
mean_age_time.head()
# The median of official time for the set of Age 
median_age_time = Boston.groupby('Age').median().set_index(np.arange(67))
median_age_time['Age'] = median_age_time.index 
median_age_time.head()
# Plotting the results

plt.figure(figsize=(12,10))

x = mean_age_time['Age']
y = mean_age_time['Official Time']

plt.plot(x, y, '.')

xx = median_age_time['Age']
yy = median_age_time['Official Time']


plt.plot(xx, yy, '.', color = 'r')


plt.xlabel("Age", fontsize=16)
plt.ylabel("Official Time (min)",fontsize=16)
plt.title("Official Time for Age",fontsize=20)
plt.legend(['Mean', 'Median'])
plt.show()
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
warnings.simplefilter("ignore", category=FutureWarning)
# Defining the dependent and independent variables 
X = mean_age_time.drop(['Official Time'], axis=1)
Y = mean_age_time['Official Time']
# Separeting the dataset in training and test datasets:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
# Constructing the model 
model = make_pipeline(PolynomialFeatures(degree=2), Ridge())
model
# Training the model with the training variables:
model.fit(X_train, Y_train)
# We show to the model the unknown test variables, in order to predict the results. 
# Then we can compare to the respective Y_test dependent variable, and check the error of the model. 
pred_test = model.predict(X_test)
# Plotting the error of the model:
plt.figure(figsize=(8,6))
#plt.scatter(pred_training, pred_training - Y_training, c = 'b', alpha = 0.5)
plt.scatter(pred_test,  pred_test - Y_test, c = 'g', alpha = 0.5)
plt.title(u"The Model Error Function", fontsize=15)
plt.show()