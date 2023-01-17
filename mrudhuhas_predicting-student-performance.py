#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix
#Reading data
data = pd.read_csv('../input/StudentsPerformance.csv')
data.head()
#Lets clean column names so it will be easier to work
data.columns = data.columns.str.strip().str.lower().str.replace(' ','_').str.replace('/','_')
data.info()
sns.pairplot(data.loc[:,['math_score','reading_score','writing_score']],height=4,aspect=1)
data.describe()
#Lets create few new features
#Total marks
data['total_marks'] = data.math_score + data.reading_score + data.writing_score
#Result
data['result'] = np.where(data['total_marks']<120, 'F', 'P')
#Percentage
data['percent'] = data.total_marks/3
#Grading - How good the student performed based on percentage
data['performance'] = pd.cut(data.percent,bins=[0,39,60,70,80,90,100],labels=['F','C','B','B+','A','A+'])
data.head()
data.info()
labels = ['male students pass','female students pass']
sizes = [data[data.result == 'P']['gender'].value_counts()[1],
         data[data.result == 'P']['gender'].value_counts()[0]]
fig1, ax1 = plt.subplots(figsize=(20,10))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
ax1.axis('equal')
plt.show()
fig1, ax1 = plt.subplots(figsize=(20,10))
ax1.pie(data[data.result == 'P']['race_ethnicity'].value_counts(),explode=(.1,.1,0,0,0),labels=['group C','group D','group B','group E','group A'],autopct='%1.1f%%',shadow=True,startangle=90)
ax1.axis('equal')
plt.show()
fig1, ax1 = plt.subplots(figsize=(20,10))
sns.countplot(data.parental_level_of_education,hue=data.result,ax=ax1)
degree_data = {
    'high school' :  'basic_education',
    'some high school' :  'basic_education',
    "associate's degree" : 'good education',
    "some college" : 'good education',
    "bachelor's degree" : 'high level education',
    "master's degree" : 'high level education'
}
data['parents_education'] = data.parental_level_of_education.map(degree_data)
fig1, ax1 = plt.subplots(figsize=(16,8))
sns.countplot(data.parents_education,ax=ax1)
sns.despine(left=True,bottom=True)
sns.set_style({'ytick.left': False,'xtick.top': False})
data[data.parents_education == 'basic_education']['result'].value_counts()
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)
ax1.pie(data[data.parents_education == 'basic_education']['result'].value_counts(),labels=['Pass','Fail'],autopct='%1.1f%%',shadow=True,startangle=90)
ax1.axis('equal')
ax1.set_title('Basic Education')
ax2.pie(data[data.parents_education == 'good education']['result'].value_counts(),labels=['Pass','Fail'],autopct='%1.1f%%',shadow=True,startangle=90)
ax2.axis('equal')
ax2.set_title('Good Education')
ax3.pie(data[data.parents_education == 'high level education']['result'].value_counts(),labels=['Pass','Fail'],autopct='%1.1f%%',shadow=True,startangle=90)
ax3.axis('equal')
ax3.set_title('High level Education')
plt.show()
fig1, ax1 = plt.subplots(figsize=(16,8))
sns.countplot(data.lunch,hue=data.result,ax=ax1)
sns.despine(left=True,bottom=True)
sns.set_style({'ytick.left': False,'xtick.top': False})
fig1, ax1 = plt.subplots(figsize=(16,8))
sns.countplot(data.test_preparation_course,ax=ax1)
sns.despine(left=True,bottom=True)
sns.set_style({'ytick.left': False,'xtick.top': False})
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.pie(data[data.test_preparation_course == 'none']['result'].value_counts(),labels=['Pass','Fail'],autopct='%1.1f%%',shadow=True,startangle=90)
ax1.axis('equal')
ax1.set_title('Test Preperation None')
ax2.pie(data[data.test_preparation_course == 'completed']['result'].value_counts(),labels=['Pass','Fail'],autopct='%1.1f%%',shadow=True,startangle=90)
ax2.axis('equal')
ax2.set_title('Test Preperation Completed')
data.head()
data_mod = data.drop(['math_score','reading_score','writing_score','total_marks','percent','performance','parents_education'],axis=1)
data_mod.head()
#Creating dummy variables
dum_r_e = pd.get_dummies(data_mod.race_ethnicity,prefix='re',dtype=int)
dum_p_l_e = pd.get_dummies(data_mod.parental_level_of_education,prefix='ple',dtype=int)

label_encoder = LabelEncoder()
data_mod.gender = label_encoder.fit_transform(data_mod.gender)
data_mod.lunch = label_encoder.fit_transform(data_mod.lunch)
data_mod.test_preparation_course = label_encoder.fit_transform(data_mod.test_preparation_course)
data_mod.result = label_encoder.fit_transform(data_mod.result)

data_mod = pd.concat([data_mod,dum_r_e,dum_p_l_e],axis=1)

data_mod = data_mod.drop(['race_ethnicity','parental_level_of_education'],axis=1)

data_mod.head()
fig, ax = plt.subplots(figsize=(16,8))
sns.heatmap(data_mod.corr(),annot=True,cmap='Blues',ax=ax)
data_mod.dtypes
x = data_mod.drop(['result'],axis=1).values
y = data_mod.loc[:,'result'].values
def backward_elimination(x_back,y,signifance_level=0.05):
    lenx = len(x_back[0])
    temp = np.zeros((1000,14)).astype(int)
    for i in range(0, lenx):
        rand_regressor_ols = sm.OLS(y, x_back).fit()
        maxPvalue = max(rand_regressor_ols.pvalues)
        adj_Rb = rand_regressor_ols.rsquared_adj.astype(float)
        if maxPvalue > signifance_level:
            for j in range(0, lenx - i):
                if (rand_regressor_ols.pvalues[j].astype(float) == maxPvalue):
                    temp[:,j] = x_back[:, j]
                    x_back = np.delete(x_back, j, 1)
                    tmp_regressor = sm.OLS(y, x_back).fit()
                    adj_Ra = tmp_regressor.rsquared_adj.astype(float)
                    if (adj_Rb >= adj_Ra):
                        x_rollback = np.hstack((x_back, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (rand_regressor_ols.summary())
                        return x_rollback
                    else:
                        continue
    rand_regressor_ols.summary()
    return x_back
x = backward_elimination(x,y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
params = {  "bootstrap":[True,False],
            "criterion":['entropy','gini'],
            "max_depth": [5,10,15],
            "max_features": ["log2", "sqrt"],
            "min_samples_split": [3,6,10],
            "min_samples_leaf": [1, 5],
            "n_estimators": [10,100,150]         
         }
rf_cla = RandomForestClassifier(random_state=1)
grid = GridSearchCV(rf_cla,param_grid=params,scoring='roc_auc',cv=10)
grid.fit(x_train,y_train)
grid.best_params_
y_pred = grid.best_estimator_.predict(x_test)
y_predtrain = grid.best_estimator_.predict(x_train)
print('Accuracy Score of test set {0}'.format(accuracy_score(y_test,y_pred)))
print('Accuracy Score of training set {0}'.format(accuracy_score(y_train,y_predtrain)))
