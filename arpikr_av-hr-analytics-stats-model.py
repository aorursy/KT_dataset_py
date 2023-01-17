pwd
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
from pandas import DataFrame
import pylab as pl
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
%matplotlib inline
train_data=pd.read_csv("../input/hranalyticsav/train_data.csv")
#test_data=pd.read_csv("C:\\Users\\ARPIT\\Desktop\\New folder\\HR Analytics\\test_data.csv")
train_data.shape
display('Train Head :',train_data.head())
#display('Test Head :',test_data.head())
print(train_data.info())
#print(test_data.info())
print("Train Data : ", train_data.nunique())
numeric_data = train_data.select_dtypes(include=[np.number])
categorical_data = train_data.select_dtypes(exclude=[np.number])
print("Numeric_Column_Count =", numeric_data.shape)
print("Categorical_Column_Count =", categorical_data.shape)
import missingno as msno
msno.matrix(train_data)
total = train_data.isnull().sum().sort_values(ascending=False)
percent_1 = train_data.isnull().sum()/train_data.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total','%'])

print(missing_data.head(5))
msno.bar(train_data)
pd.options.display.float_format = '{:.4f}'.format
train_data.describe().T
my_corr=train_data.corr()
plt.figure(figsize=(18,8))
sns.set(font_scale=0.8)
sns.heatmap(my_corr, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},linewidth=0.8)
plt.show()
cor_target = abs(my_corr["is_promoted"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
relevant_features
plt.figure(figsize=(10,6))
print(train_data['is_promoted'].value_counts())
sns.countplot(x='is_promoted',data=train_data)
plt.figure(figsize=(15,8))
train_data.boxplot(column=['age','previous_year_rating','length_of_service','KPIs_met >80%','awards_won?','avg_training_score'], grid=False)
plt.figure(figsize=(15,8))
sns.distplot(train_data['age'])
plt.show() 
sns.set_style("white")
plt.figure (figsize=(8,6))
sns.distplot(train_data['length_of_service'], axlabel = 'Service Length')
promoted= 'promoted'
not_promoted = 'not promoted'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 6))
female = train_data[train_data['gender']=='f']
male = train_data[train_data['gender']=='m']
ax = sns.distplot(female[female['is_promoted']==1].age.dropna(), bins=18, label = promoted, ax = axes[0], kde =False)
ax = sns.distplot(female[female['is_promoted']==0].age.dropna(), bins=40, label = not_promoted, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(male[male['is_promoted']==1].age.dropna(), bins=18, label = promoted, ax = axes[1], kde = False)
ax = sns.distplot(male[male['is_promoted']==0].age.dropna(), bins=40, label = not_promoted, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')
FacetGrid = sns.FacetGrid(train_data, row='education', size=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'recruitment_channel','is_promoted', 'gender', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()
#https://seaborn.pydata.org/tutorial/categorical.html
size = [30446, 23220, 1142]
colors = ['yellow', 'red', 'lightgreen']
labels = "Others", "Sourcing", "Reffered"

my_circle = plt.Circle((0, 0), 0.7, color = 'white')

plt.rcParams['figure.figsize'] = (9, 9)
plt.pie(size, colors = colors, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('Showing share of different Recruitment Channels', fontsize = 30)
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.legend()
plt.show()

pd.crosstab([train_data.education],train_data.is_promoted,margins=True).style.background_gradient(cmap='Wistia')
grid = sns.FacetGrid(train_data, col='is_promoted', row='no_of_trainings', size=3.5, aspect=1.6)
grid.map(plt.hist, 'age', alpha=.5, bins=20)
grid.add_legend();
grid = sns.FacetGrid(train_data, col='is_promoted', row='previous_year_rating', size=3.5, aspect=1.6)
grid.map(plt.hist, 'age', alpha=.5, bins=20)
grid.add_legend();
# checking the distribution of the avg_training score of the Employees
plt.rcParams['figure.figsize'] = (15, 7)
sns.distplot(train_data['avg_training_score'], color = 'blue')
plt.title('Distribution of Training Score among the Employees', fontsize = 30)
plt.xlabel('Average Training Score', fontsize = 20)
plt.ylabel('count')
plt.show()
size = [53538, 1270]
colors = ['magenta', 'brown']
labels = "Awards Won", "NO Awards Won"

my_circle = plt.Circle((0, 0), 0.7, color = 'white')

plt.rcParams['figure.figsize'] = (9, 9)
plt.pie(size, colors = colors, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('Showing a Percentage of employees who won awards', fontsize = 30)
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.legend()
plt.show()
size = [35517, 19291]
labels = "Not Met KPI > 80%", "Met KPI > 80%"
colors = ['violet', 'grey']
explode = [0, 0.1]

plt.rcParams['figure.figsize'] = (8, 8)
plt.pie(size, labels = labels, colors = colors, explode = explode, shadow = True, autopct = "%.2f%%")
plt.title('A Pie Chart Representing Gap in Employees in terms of KPI', fontsize = 30)
plt.axis('off')
plt.legend()
plt.show()

pd.crosstab( train_data.is_promoted,train_data.region,margins=True).style.background_gradient(cmap='Wistia')

# checking dependency of different regions in promotion
data = pd.crosstab(train_data['region'], train_data['is_promoted'])
data.div(data.sum(1).astype('float'), axis = 0).plot(kind = 'bar', stacked = True, figsize = (20, 8), color = ['lightblue', 'purple'])

plt.title('Dependency of Regions in determining Promotion of Employees', fontsize = 30)
plt.xlabel('Different Regions of the Company', fontsize = 20)
plt.legend()
plt.show()
# dependency of awards won on promotion

data = pd.crosstab(train_data['awards_won?'], train_data['is_promoted'])
data.div(data.sum(1).astype('float'), axis = 0).plot(kind = 'bar', stacked = True, figsize = (10, 8), color = ['magenta', 'purple'])

plt.title('Dependency of Awards in determining Promotion', fontsize = 30)
plt.xlabel('Awards Won or Not', fontsize = 20)
plt.legend()
plt.show()

# scatter plot between average training score and is_promoted

data = pd.crosstab(train_data['avg_training_score'], train_data['is_promoted'])
data.div(data.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (20, 9), color = ['darkred', 'lightgreen'])

plt.title('Looking at the Dependency of Training Score in promotion', fontsize = 30)
plt.xlabel('Average Training Scores', fontsize = 15)
plt.legend()
plt.show()
train_data['education'].fillna(train_data['education'].mode()[0], inplace = True)
train_data['previous_year_rating'].fillna(train_data['previous_year_rating'].mode()[0], inplace = True)
train_data['previous_year_rating'].astype(int)
axes = sns.factorplot('previous_year_rating','is_promoted',data=train_data, aspect = 2.5, )
train_data.groupby(['department','recruitment_channel']).size().unstack().plot(figsize=(15,8),kind='barh',stacked=True)
plt.show()
axes = sns.factorplot('no_of_trainings','is_promoted',data=train_data, aspect = 2.5, )
plt.figure(figsize=(12,5))
sns.countplot('recruitment_channel',hue='is_promoted',data=train_data).set_title('Promotion_Recruitment Channel')
plt.figure(figsize=(12,5))
sns.countplot(x='education', hue='is_promoted', data=train_data)
plt.figure(figsize=(12,5))
sns.countplot(x='department', hue='KPIs_met >80%', data=train_data)
#https://seaborn.pydata.org/generated/seaborn.countplot.html
#train_data.groupby(['region']).sum().plot(kind='pie', y='avg_training_score',startangle=90,figsize=(15,10), autopct='%1.1f%%')
#https://plotly.com/python/pie-charts/
import plotly.graph_objects as go
labels = train_data['region']
values = train_data['avg_training_score']
fig = go.Figure(data=[go.Pie(labels=labels, values=values,hole=.5)])
fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
fig.show()
import plotly.graph_objects as go
labels = train_data['department']
values = train_data['avg_training_score']
fig = go.Figure(data=[go.Pie(labels=labels, values=values,hole=0.0)])
fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
fig.show()
#plt.figure(figsize=(12,7))
#train_data['department'].value_counts().plot(kind='pie')
#train_data['previous_year_rating'].fillna(train_data['previous_year_rating'].mode()[0], inplace = True)
sns.pairplot(train_data);
a = sns.FacetGrid(train_data, hue = 'is_promoted', aspect=4 )
a.map(sns.kdeplot, 'age', shade= True )
a.set(xlim=(0 ,train_data['age'].max()))
a.add_legend()
pd.get_dummies(train_data['gender'], prefix='G')
train_data = pd.concat([train_data, pd.get_dummies(train_data['gender'], prefix='G')], axis=1)
train_data.drop(['gender','G_f'],axis=1,inplace=True)
train_data.head()
pd.get_dummies(train_data['recruitment_channel'], prefix='R')
train_data = pd.concat([train_data, pd.get_dummies(train_data['recruitment_channel'], prefix='R')], axis=1)
train_data.drop(['recruitment_channel','R_other'],axis=1,inplace=True)
train_data.head()
pd.get_dummies(train_data['region'], prefix='Re')
train_data = pd.concat([train_data, pd.get_dummies(train_data['region'], prefix='Re')], axis=1)
train_data.drop(['region','Re_region_8'],axis=1,inplace=True)
pd.get_dummies(train_data['department'], prefix='Dep')
train_data = pd.concat([train_data, pd.get_dummies(train_data['department'], prefix='Dep')], axis=1)
train_data.drop(['department','Dep_Technology'],axis=1,inplace=True)
replace={"Master's & above":3,"Bachelor's":2,"Below Secondary":1}
train_data['education']=train_data['education'].replace(replace)
train_data=train_data.drop(['employee_id'],axis=1,inplace=False)
train_data.head()
X = train_data.drop(['is_promoted'],axis = 1)
Y = train_data['is_promoted'] 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=123)
import statsmodels.api as sm
X_train = sm.add_constant(X_train)   # Adding Constant Term to X_Train 
y_train.head()
logit = sm.GLM(y_train, X_train, family=sm.families.Binomial())   #Fit Logistic Regression.
result = logit.fit()
print(result.summary())
print(result.summary2())
result.params
np.exp(result.params)
result.deviance
result.aic
X_train.drop(["Re_region_20","Re_region_19","Re_region_31"], axis=1, inplace=True)    #Dropping the Fare Column as it has P-value>0.05
print(sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit().summary2())
X_train.drop(["Re_region_5","G_m","R_referred"], axis=1, inplace=True)    #Dropping the Fare Column as it has P-value>0.05
print(sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit().summary2())
X_train.drop(["Re_region_33","Re_region_6","Re_region_26","Re_region_24","Re_region_21","Re_region_18","Re_region_16","Re_region_14","Re_region_15","Re_region_12","Re_region_11","Re_region_10","Re_region_1","R_sourcing"], axis=1, inplace=True)    #Dropping the Fare Column as it has P-value>0.05
print(sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit().summary2())
X_train.drop(["Re_region_13","Re_region_3","Re_region_30"], axis=1, inplace=True)    #Dropping the Fare Column as it has P-value>0.05
print(sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit().summary2())
result1= sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()
result1.deviance
result1.aic
X_test1 = sm.add_constant(X_test[["education","no_of_trainings","age","previous_year_rating","length_of_service",
                                 "KPIs_met >80%","awards_won?","avg_training_score","Re_region_17","Re_region_2",
                                 "Re_region_22","Re_region_23","Re_region_25","Re_region_27","Re_region_28","Re_region_29",
                                 "Re_region_32","Re_region_34","Re_region_4","Re_region_7","Re_region_9","Dep_Analytics",
                                 "Dep_Finance","Dep_HR","Dep_Legal","Dep_Operations","Dep_Procurement","Dep_R&D","Dep_Sales & Marketing"]])
probabilites = result1.predict(X_test1)
probabilites.head()
predicted_classes = probabilites.map(lambda x: 1 if x > 0.5 else 0)
accuracy = sum(predicted_classes == y_test) / len(y_test)
accuracy = sum(predicted_classes == y_test) / len(y_test)
accuracy