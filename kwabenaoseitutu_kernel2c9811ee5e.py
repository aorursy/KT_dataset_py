# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
my_filepath = "../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv"
my_data = pd.read_csv(my_filepath, index_col="sl_no")
my_data.head()
my_data.shape
my_data.info
my_data.describe()
my_data.describe(include=object)
missing_values = my_data.isnull()

missing_values.head()
for column in missing_values.columns.values.tolist():

    print(column)

    print (missing_values[column].value_counts())

    print("")
my_data["salary"]=my_data["salary"].fillna(0)
my_data.columns
print(my_data.status.value_counts(),"\n",

my_data.status.value_counts(normalize=True))


my_data.status.value_counts().plot.bar(title='Count of employed and unemployed candidates')
plt.figure(1) 

plt.subplot(131)

my_data.gender.value_counts().plot.bar(figsize=(10,8), title="Gender")

plt.subplot(132)

my_data.ssc_b.value_counts().plot.bar(figsize=(10,8),title="S S Board of Education")

plt.subplot(133)

my_data.hsc_b.value_counts().plot.bar(figsize=(10,8), title ="Higher School Board of Education")

plt.show()

plt.figure() 

my_data.hsc_s.value_counts().plot.bar(figsize=(8,6), title="Count of students per Specialization in Higher Secondary")

plt.show()
my_data.degree_t.value_counts().plot.bar(figsize=(8,6),title="Count of students per UnderGrad Degree type")
my_data.workex.value_counts().plot.bar(figsize=(8,6), title ="Count of students per Work experience")
my_data.specialisation.value_counts().plot.bar(figsize=(8,6), title ="Count of students per Post Graduation(MBA)- Specialization")
plt.figure(figsize=(16,6))

plt.subplot(141)

plt.title("Distribution of Secondary School Performance")

sns.distplot(a=my_data.ssc_p,kde=False)

plt.xlabel("Secondary School Exam Percentage scored")

plt.ylabel("Number of candidates ")

plt.subplot(142)

sns.boxplot(x=my_data.ssc_p)

plt.subplot(143)

sns.boxplot(x=my_data.status, y=my_data.ssc_p)

plt.subplot(144)

sns.swarmplot(x=my_data.status, y=my_data.ssc_p)

plt.show()


plt.figure(figsize=(16,6))

plt.subplot(141)

plt.title("Distribution of High School Performance")

sns.distplot(a=my_data.hsc_p,kde=False)

plt.xlabel("High School Exam Percentage scored")

plt.ylabel("Number of candidates ")

plt.subplot(142)

sns.boxplot(x=my_data.hsc_p)

plt.subplot(143)

sns.boxplot(x=my_data.status, y=my_data.hsc_p)

plt.subplot(144)

sns.swarmplot(x=my_data.status, y=my_data.hsc_p)

plt.show()
plt.figure(figsize=(16,6))

plt.subplot(141)

plt.title("Distribution of UnderGrad Performance")

sns.distplot(a=my_data.degree_p,kde=False)

plt.xlabel("UnderGrad Exam Percentage scored")

plt.ylabel("Number of candidates")

plt.subplot(142)

sns.boxplot(x=my_data.degree_p)

plt.subplot(143)

sns.boxplot(x=my_data.status, y=my_data.degree_p)

plt.subplot(144)

sns.swarmplot(x=my_data.status, y=my_data.degree_p)

plt.show()
plt.figure(figsize=(16,6))

plt.subplot(141)

plt.title("Employability test percentage (conducted by college)")

sns.distplot(a=my_data.etest_p,kde=False)

plt.xlabel("Employability Exam score")

plt.ylabel("Number of candidates")

plt.subplot(142)

sns.boxplot(x=my_data.etest_p)

plt.subplot(143)

sns.boxplot(x=my_data.status, y=my_data.etest_p)

plt.subplot(144)

sns.swarmplot(x=my_data.status, y=my_data.etest_p)

plt.show()
plt.figure(figsize=(16,6))

plt.subplot(141)

plt.title("Distribution of MBA percentage")

sns.distplot(a=my_data.mba_p,kde=False)

plt.subplot(142)

sns.boxplot(x=my_data.mba_p)

plt.subplot(143)

plt.title("Distribution of MBA percentage per placement status")

sns.boxplot(x=my_data.status, y=my_data.mba_p)

plt.subplot(144)

sns.swarmplot(x=my_data.status, y=my_data.mba_p)

plt.show()
plt.figure(figsize=(16,6))

plt.subplot(141)

plt.title("Distribution of salary")

sns.distplot(a=my_data.salary,kde=False)

plt.subplot(142)

sns.boxplot(x=my_data.salary)

plt.subplot(143)

plt.title("Distribution of salary per placement status")

sns.boxplot(x=my_data.gender, y=my_data.salary)

plt.subplot(144)

sns.swarmplot(x=my_data.gender, y=my_data.salary)



plt.show()
Gender=pd.crosstab(my_data['gender'],my_data['status']) 

Gender.div(Gender.sum(1), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))



ssbe = pd.crosstab(my_data.ssc_b,my_data.status)

ssbe.div(ssbe.sum(1), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))



hsbe = pd.crosstab(my_data.hsc_b,my_data.status)

hsbe.div(hsbe.sum(1), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

ssbe = pd.crosstab(my_data.ssc_b,my_data.status)

ssbe.div(ssbe.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
hscs = pd.crosstab(my_data.hsc_s,my_data.status)

hscs.div(hscs.sum(1),axis=0).plot(kind="bar", stacked=True, figsize=(4,4))



degreeT = pd.crosstab(my_data.degree_t,my_data.status)

degreeT.div(degreeT.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))



workX = pd.crosstab(my_data.workex,my_data.status)

workX.div(workX.sum(1), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))



specialty = pd.crosstab(my_data.specialisation, my_data.status)

specialty.div(specialty.sum(1), axis=0).plot(kind="bar", stacked="True", figsize=(4,4))
mycorr = my_data[['ssc_p','hsc_p','degree_p','etest_p','mba_p','salary']].corr()

sns.heatmap(mycorr,annot=True)
sns.lmplot(x="degree_p", y="mba_p", hue="status", data=my_data)
sns.lmplot(x="ssc_p", y="hsc_p", hue="status", data=my_data)
sns.lmplot(x="etest_p", y="mba_p", hue="status", data=my_data)
sns.lmplot(x="mba_p", y="salary", hue="specialisation", data=my_data)
#Separating the the independent variable(X) and the target variable(y) from the dataset

X = my_data.drop('status',axis=1)

y = my_data.status
#OneHotEncoding will be deployed to change encode categorical variables with more than two unique items

degreedummy = pd.get_dummies(X.degree_t)

hscsdummy = pd.get_dummies(X.hsc_s)
X = pd.concat([X, degreedummy], axis=1)

X = pd.concat([X, hscsdummy], axis=1)
# drop original column of onehotencoded columns from X

X.drop("degree_t", axis = 1, inplace=True)

X.drop("hsc_s", axis = 1, inplace=True)



#Droping Salary since unemployed candidates automatically have 0 salary

X.drop("salary", axis = 1, inplace=True)
#LabelEncoder will be deployed to encode variables with two unique items

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

X['gender'] = le.fit_transform(X.gender)

X['ssc_b'] = le.fit_transform(X.ssc_b)

X['hsc_b'] = le.fit_transform(X.hsc_b)

X['workex'] = le.fit_transform(X.workex)

X['specialisation'] = le.fit_transform(X.specialisation)

y = le.fit_transform(y)
#Standardizing the data to ensure all distributions are normal and also suppress outliers

from sklearn import preprocessing

X['etest_p'] = preprocessing.scale(X.etest_p)

X['degree_p'] = preprocessing.scale(X.degree_p)

X['mba_p'] = preprocessing.scale(X.mba_p)

X['ssc_p'] = preprocessing.scale(X.ssc_p)

X['hsc_p'] = preprocessing.scale(X.hsc_p)

X['degree_p'] = preprocessing.scale(X.degree_p)
X.tail()
X.shape, y.shape
#Split data for training & validation

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state = 1)
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
#Decision Tree





dmodel = DecisionTreeClassifier()



dmodel.fit(X_train,y_train)



y_predx = dmodel.predict(X_test)
accuracy_score(y_predx,y_test)
print(classification_report(y_test,y_predx))
#confusion matrix: Idea for a yes/no predictions

print(confusion_matrix(y_test,y_predx))



sns.heatmap(confusion_matrix(y_test,y_predx),annot=True,lw =2,cbar=False)

plt.ylabel("True Values")

plt.xlabel("Predicted Values")

plt.title("CONFUSSION MATRIX VISUALIZATION")

plt.show()
importances=pd.Series(dmodel.feature_importances_, index=X.columns) 

importances.plot(kind='barh', figsize=(8,6))
#Random Forest

rtmodel=RandomForestClassifier()



rtmodel.fit(X_train,y_train)

y_predr = rtmodel.predict(X_test)
accuracy_score(y_predr,y_test)
#confusion matrix: Idea for a yes/no predictions

print(confusion_matrix(y_test,y_predr))



sns.heatmap(confusion_matrix(y_test,y_predr),annot=True,lw =2,cbar=False)

plt.ylabel("True Values")

plt.xlabel("Predicted Values")

plt.title("CONFUSSION MATRIX VISUALIZATION")

plt.show()
print(classification_report(y_test,y_predr))
importances=pd.Series(rtmodel.feature_importances_, index=X.columns) 

importances.plot(kind='barh', figsize=(8,6))
#Boost for random forest



from sklearn.model_selection import GridSearchCV
# Provide range for max_depth from 1 to 20 with an interval of 2 and from 1 to 200 with an interval of 20 for n_estimators 

paramgrid = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}

grid_search=GridSearchCV(RandomForestClassifier(random_state=1),paramgrid)
# Fit the grid search model 

grid_search.fit(X_train,y_train)
# Estimating the optimized value 

grid_search.best_estimator_
RFCmodel = RandomForestClassifier(max_depth=3, n_estimators=141, random_state=1)

RFCmodel.fit(X_train,y_train)

y_predrfc = RFCmodel.predict(X_test)
accuracy_score(y_predrfc,y_test)
#confusion matrix: Idea for a yes/no predictions

print(confusion_matrix(y_test,y_predrfc))



sns.heatmap(confusion_matrix(y_test,y_predrfc),annot=True,lw =2,cbar=False)

plt.ylabel("True Values")

plt.xlabel("Predicted Values")

plt.title("CONFUSSION MATRIX VISUALIZATION")

plt.show()
print(classification_report(y_test,y_predrfc))
importances=pd.Series(RFCmodel.feature_importances_, index=X.columns) 

importances.plot(kind='barh', figsize=(8,6))
#pip install xgboost
from xgboost import XGBClassifier
# fit model no training data

XGmodel = XGBClassifier()

XGmodel.fit(X_train, y_train)

y_predXG = XGmodel.predict(X_test)
accuracy_score(y_predXG,y_test)
#confusion matrix: Idea for a yes/no predictions

print(confusion_matrix(y_test,y_predXG))



sns.heatmap(confusion_matrix(y_test,y_predXG),annot=True,lw =2,cbar=False)

plt.ylabel("True Values")

plt.xlabel("Predicted Values")

plt.title("CONFUSSION MATRIX VISUALIZATION")

plt.show()
print(classification_report(y_test,y_predXG))
importances=pd.Series(XGmodel.feature_importances_, index=X.columns) 

importances.plot(kind='barh', figsize=(8,6))
parameters = {

    'max_depth': range (2, 10, 1),

    'n_estimators': range(60, 220, 40),

    'learning_rate': [0.1, 0.01, 0.05]

}
estimator = XGBClassifier(

    objective= 'binary:logistic',

    nthread=4,

    seed=42

)
grid_search = GridSearchCV(

    estimator=estimator,

    param_grid=parameters,

    scoring = 'roc_auc',

    n_jobs = 10,

    cv = 10,

    verbose=True

)
# Fit the grid search model 

grid_search.fit(X_train,y_train)
# Estimating the optimized value 

grid_search.best_estimator_
XGmodel2 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

              importance_type='gain', interaction_constraints='',

              learning_rate=0.1, max_delta_step=0, max_depth=5,

              min_child_weight=1, monotone_constraints='()',

              n_estimators=180, n_jobs=4, nthread=4, num_parallel_tree=1,

              random_state=42, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,

              seed=42, subsample=1, tree_method='exact', validate_parameters=1,

              verbosity=None)

XGmodel2.fit(X_train,y_train)

y_predXG2 = XGmodel2.predict(X_test)
accuracy_score(y_predXG2,y_test)


print(confusion_matrix(y_test,y_predXG2))



sns.heatmap(confusion_matrix(y_test,y_predXG2),annot=True,lw =2,cbar=False)

plt.ylabel("True Values")

plt.xlabel("Predicted Values")

plt.title("CONFUSSION MATRIX VISUALIZATION")

plt.show()
print(classification_report(y_test,y_predXG2))
importances=pd.Series(XGmodel2.feature_importances_, index=X.columns) 

importances.plot(kind='barh', figsize=(8,6))
from lightgbm import LGBMClassifier
# fit the model on the whole dataset

LGmodel = LGBMClassifier()

LGmodel.fit(X_train, y_train)

y_predL = LGmodel.predict(X_test)
accuracy_score(y_predL,y_test)
#confusion matrix: Idea for a yes/no predictions

print(confusion_matrix(y_test,y_predL))



sns.heatmap(confusion_matrix(y_test,y_predL),annot=True,lw =2,cbar=False)

plt.ylabel("True Values")

plt.xlabel("Predicted Values")

plt.title("CONFUSSION MATRIX VISUALIZATION")

plt.show()
print(classification_report(y_test,y_predL))
importances=pd.Series(LGmodel.feature_importances_, index=X.columns) 

importances.plot(kind='barh', figsize=(8,6))
#pip install catboost
from catboost import CatBoostClassifier
CatModel = CatBoostClassifier()

CatModel.fit(X_train,y_train)

y_predCat = CatModel.predict(X_test)
accuracy_score(y_predCat,y_test)
#confusion matrix: Idea for a yes/no predictions

print(confusion_matrix(y_test,y_predCat))



sns.heatmap(confusion_matrix(y_test,y_predCat),annot=True,lw =2,cbar=False)

plt.ylabel("True Values")

plt.xlabel("Predicted Values")

plt.title("CONFUSSION MATRIX VISUALIZATION")

plt.show()
print(classification_report(y_test,y_predCat))
importances=pd.Series(CatModel.feature_importances_, index=X.columns) 

importances.plot(kind='barh', figsize=(8,6))
from sklearn.ensemble import VotingClassifier
eclf = VotingClassifier(estimators=[('dt', dmodel), ('rf1', rtmodel), ('rf2', RFCmodel),('cat', CatModel), ('lgb', LGmodel),('xgb', XGmodel),('xgb2', XGmodel2)], voting='soft', weights=[1, 1, 1, 1, 1, 1, 1])

eclf.fit(X_train, y_train)

y_pred_ens= eclf.predict(X_test)
accuracy_score(y_pred_ens,y_test)
#confusion matrix: Idea for a yes/no predictions

print(confusion_matrix(y_test,y_pred_ens))



sns.heatmap(confusion_matrix(y_test,y_pred_ens),annot=True,lw =2,cbar=False)

plt.ylabel("True Values")

plt.xlabel("Predicted Values")

plt.title("CONFUSSION MATRIX VISUALIZATION")

plt.show()
print(classification_report(y_test,y_pred_ens))
eclf2 = VotingClassifier(estimators=[('dt', dmodel), ('rf1', rtmodel), ('rf2', RFCmodel),('cat', CatModel), ('lgb', LGmodel),('xgb', XGmodel),('xgb2', XGmodel2)], voting='hard', weights=[1, 1, 1, 1, 1, 1, 1])

eclf2.fit(X_train, y_train)
y_pred_ens2 = eclf2.predict(X_test)
accuracy_score(y_pred_ens2,y_test)
#confusion matrix: Idea for a yes/no predictions

print(confusion_matrix(y_test,y_pred_ens2))



sns.heatmap(confusion_matrix(y_test,y_pred_ens2),annot=True,lw =2,cbar=False)

plt.ylabel("True Values")

plt.xlabel("Predicted Values")

plt.title("CONFUSSION MATRIX VISUALIZATION")

plt.show()
print(classification_report(y_test,y_pred_ens2))