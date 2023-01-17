# Load necessary library

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.ticker import StrMethodFormatter

import matplotlib.ticker as mtick

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report



import statsmodels.api as sm

import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn')

%matplotlib inline



# set default plot size

plt.rcParams["figure.figsize"] = (15,8)
# Load and preview data 

recruit = pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")

recruit.head()
# drop id column

recruit.drop('sl_no',axis=1,inplace=True)

recruit.shape
# Summary Statistics

recruit.describe()
# Check each column for nas

recruit.isnull().sum()
sns.pairplot(recruit.drop('salary',axis=1),hue = 'status')



# gender             0

# ssc_p              0

# ssc_b              0

# hsc_p              0

# hsc_b              0

# hsc_s              0

# degree_p           0

# degree_t           0

# workex             0

# etest_p            0

# specialisation     0

# mba_p              0

# status             0

# salary            67
recruit.groupby(["gender","status"]).size().unstack()
recruit.groupby(["gender","status"]).size().groupby(level=0).apply(

    lambda x: 100 * x / x.sum()

).unstack().plot(kind='bar',stacked=True)



plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

plt.legend(loc = 'upper right',title = 'Status')

plt.show()



# most of males are placed job than female
recruit.groupby('status').mean()
recruit_numeric = recruit[['ssc_p','hsc_p','degree_p','etest_p','mba_p','status']]



recruit_numeric_melt = pd.melt(recruit_numeric,id_vars='status',

                               value_vars =['ssc_p','hsc_p','degree_p','etest_p','mba_p'])

recruit_numeric_melt.head()
sns.boxplot(x="variable", y="value",

            hue="status", data=recruit_numeric_melt)
# then will look at all the categorical variables



# column description 

# ssc_b              Board of Education- Central/ Others

# hsc_b              Board of Education- Central/ Others

# hsc_s              Specialization in Higher Secondary Education

# degree_t           Under Graduation(Degree type)- Field of degree education

# workex             Work Experience

# specialisation     Post Graduation(MBA)- Specialization

# status             Status of placement- Placed/Not placed

# salary             Salary offered by corporate to candidates





# Board of Education - 10th grade

recruit.groupby(["ssc_b","status"]).size().groupby(level=0).apply(

    lambda x: 100 * x / x.sum()

).unstack().plot(kind='bar',stacked=True)



plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

plt.legend(loc = 'upper right',title = 'Board of Education')

plt.show()



# central and others almost no difference for secondary education board of education
# Board of Education - 12th grade

recruit.groupby(["hsc_b","status"]).size().groupby(level=0).apply(

    lambda x: 100 * x / x.sum()

).unstack().plot(kind='bar',stacked=True)



plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

plt.legend(loc = 'upper right',title = 'Board of Education')

plt.show()



# similarly central and others almost no difference for secondary education board of education
# Specialization in Higher Secondary Education

recruit.groupby(["hsc_s","status"]).size().groupby(level=0).apply(

    lambda x: 100 * x / x.sum()

).unstack().plot(kind='bar',stacked=True)



plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

plt.legend(loc = 'upper right',title = 'Higher Education Specialization')

plt.show()



# commerce and science are more likely to get placed
# Under Graduation(Degree type)- Field of degree education

recruit.groupby(["degree_t","status"]).size().groupby(level=0).apply(

    lambda x: 100 * x / x.sum()

).unstack().plot(kind='bar',stacked=True)



plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

plt.legend(loc = 'upper right',title = 'Degree type')

plt.show()



# for undergraduate degrees, comm/management and sci/tech are more likely to get placed
# Work Experience

recruit.groupby(["workex","status"]).size().groupby(level=0).apply(

    lambda x: 100 * x / x.sum()

).unstack().plot(kind='bar',stacked=True)



plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

plt.legend(loc = 'upper right',title = 'Work experience')

plt.show()



# having working experience is more likely to get placed and it has the most influence by comparing the graphs
# Post Graduation(MBA)- Specialization

recruit.groupby(["specialisation","status"]).size().groupby(level=0).apply(

    lambda x: 100 * x / x.sum()

).unstack().plot(kind='bar',stacked=True)



plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

plt.legend(loc = 'upper right',title = 'specialisation')

plt.show()



# mrkt/finance are more likely to get placed than mrkt/hr
# transfer categorical vaeiables to dummy variables



recruit.loc[recruit['gender'] == 'M', 'gender'] = 1.0

recruit.loc[recruit['gender'] == 'F', 'gender'] = 0.0



recruit.loc[recruit['status'] == 'Placed', 'status'] = 1

recruit.loc[recruit['status'] == 'Not Placed', 'status'] = 0



recruit.loc[recruit['workex'] == 'Yes', 'workex'] = 1.0

recruit.loc[recruit['workex'] == 'No', 'workex'] = 0.0





categorical_var = ['ssc_b','hsc_b','hsc_s','degree_t','specialisation']





# create dummy variables for all the other categorical variables



for variable in categorical_var:

# #     fill missing data

#     recruit[variable].fillna('Missing',inplace=True)

#     create dummy variables for given columns

    dummies = pd.get_dummies(recruit[variable],prefix=variable)

#     update data and drop original columns

    recruit = pd.concat([recruit,dummies],axis=1)

    recruit.drop([variable],axis=1,inplace=True)





recruit.head()
# Create separate dataset for placed status

# use this for further regression analysis

recruit_placed = recruit[recruit['status'] == 1].drop('status',axis = 1)

recruit_placed.head()
x = recruit.drop(['status','salary'], axis=1)

y = recruit['status'].astype(float)



# split train and test dataset

train_x, test_x, train_y, test_y = train_test_split(x,y , test_size=0.3, random_state=42)



print(train_x.shape)

print(train_y.shape)
rf_regressor = RandomForestRegressor(100, oob_score=True,

                                     n_jobs=-1, random_state=42)

rf_regressor.fit(train_x,train_y)

print('Score: ', rf_regressor.score(train_x,train_y))
feature_importance = pd.Series(rf_regressor.feature_importances_,index=x.columns)

feature_importance = feature_importance.sort_values()

feature_importance.plot(kind='barh')
# parameter tunning

# # of trees trained parameter tunning



results = []

n_estimator_options = [30,50,100,200,500,1000,2000]



for trees in n_estimator_options:

    model = RandomForestRegressor(trees,oob_score=True,n_jobs=-1,random_state=42)

    model.fit(x,y)

    print(trees," trees")

    score = model.score(train_x,train_y)

    print(score)

    results.append(score)

    print("")



pd.Series(results,n_estimator_options).plot()
# max number of features parameter tunning

results = []

max_features_options = ['auto',None,'sqrt','log2',0.9,0.2]



for max_features in max_features_options:

    model = RandomForestRegressor(n_estimators=200,oob_score=True,n_jobs=-1,

                                  random_state=42,max_features=max_features)

    model.fit(x,y)

    print(max_features," option")

    score = model.score(train_x,train_y)

    print(score)

    results.append(score)

    print("")



pd.Series(results,max_features_options).plot(kind='barh')
# min sample leaf parameter tunning

results = []

min_sample_leaf_option = [1,2,3,4,5,6,7,8,9,10]



for min_sample_leaf in min_sample_leaf_option:

    model = RandomForestRegressor(n_estimators=200,oob_score=True,n_jobs=-1,

                                  random_state=42,max_features='sqrt',

                                  min_samples_leaf=min_sample_leaf)

    model.fit(x,y)

    print(min_sample_leaf," min samples")

    score = model.score(train_x,train_y)

    print(score)

    results.append(score)

    print("")



pd.Series(results,min_sample_leaf_option).plot()
rf_regressor = RandomForestRegressor(200, oob_score=True,max_features='sqrt',

                                     n_jobs=-1, random_state=42,min_samples_leaf=1)

rf_regressor.fit(x,y)

print('Score: ', rf_regressor.score(train_x,train_y))
pred_y = rf_regressor.predict(test_x)



print(test_y[:10])

print(pred_y[:10])
rf_classifier = RandomForestClassifier(200, oob_score=True,

                                     n_jobs=-1, random_state=42)

rf_classifier.fit(train_x, train_y)
pred_y = rf_classifier.predict(test_x)
rf_classifier.score(test_x, test_y)
mat = confusion_matrix(test_y,pred_y)

sns.heatmap(mat, square=True, annot=True, cbar=False) 

plt.xlabel('predicted value')

plt.ylabel('true value')
print(classification_report(test_y, pred_y))
lr_model = LogisticRegression()

lr_model.fit(train_x,train_y)
lr_model.score(test_x, test_y)
pred_y = lr_model.predict(test_x)

mat = confusion_matrix(test_y,pred_y)

sns.heatmap(mat, square=True, annot=True, cbar=False) 

plt.xlabel('predicted value')

plt.ylabel('true value')
print(classification_report(test_y, pred_y))
lr_coef = pd.DataFrame({"Coefficients":lr_model.coef_[0]},index = x.columns.tolist())

lr_coef = lr_coef.sort_values(by = 'Coefficients')

lr_coef
lr_coef.plot(kind='barh')
recruit_placed.head()
sns.pairplot(recruit_placed[['ssc_p','hsc_p','degree_p','etest_p','mba_p','salary']])
recruit_placed[['ssc_p','hsc_p','degree_p','etest_p','mba_p','salary']].corr()
var = ['ssc_p','hsc_p','degree_p','etest_p','mba_p','gender','workex']

x = recruit_placed.loc[:,var]

# x = recruit_placed.loc[:,recruit_placed.columns != 'salary']

y = recruit_placed.loc[:,recruit_placed.columns == 'salary']

x.head()
train_x, test_x, train_y, test_y = train_test_split(x,y , test_size=0.2, random_state=42)



print(train_x.shape)

print(test_x.shape)
linear_model = sm.OLS(train_y,train_x.astype(float))

results = linear_model.fit()

results.params
print(results.summary())
pred_y = results.predict(test_x)

# print(pred_y[:10])

# print(test_y[:10])



col = ['actual','prediction']



prediction = pd.concat([test_y,pred_y],axis=1)

prediction.columns = col

prediction
_, ax = plt.subplots()



ax.scatter(x = range(0, test_y.size), y=test_y, c = 'blue', label = 'Actual', alpha = 0.3)

ax.scatter(x = range(0, pred_y.size), y=pred_y, c = 'red', label = 'Predicted', alpha = 0.3)



plt.title('Actual and predicted values')

plt.xlabel('Observations')

plt.ylabel('Salary')

plt.legend()

plt.show()