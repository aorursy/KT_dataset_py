import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix



from imblearn.over_sampling import SVMSMOTE

from imblearn.under_sampling import RandomUnderSampler



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier



from yellowbrick.model_selection import FeatureImportances
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/income-classification/income_evaluation.csv')
df.head()
df.columns
df.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain',

              'capital-loss','hours-per-week','native-country','income']



df.columns = df.columns.str.replace('-','_')
df.head()
df.shape
df.info()
df.describe().T
categorical = [var for var in df.columns if df[var].dtype=='O']



print(categorical)
df[categorical].describe()
df.isnull().sum()
data = df.copy()
sns.countplot(data['income'])
sns.distplot(data['age'])
labels = ['10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90']

bins = [10,20,30,40,50,60,70,80,90]

freq_df = data.groupby(pd.cut(data['age'],bins = bins,labels = labels)).size()

freq_df = freq_df.reset_index(name = 'count')
freq_df
plt.bar(freq_df['age'],freq_df['count'])
data['workclass'].value_counts()
data['workclass'] = data.workclass.str.replace('?','Unknown')
sns.countplot(data['workclass'])

plt.xticks(rotation = 90)
sns.distplot(data['fnlwgt'])
data['education'].value_counts()
sns.countplot(data['education'])

plt.xticks(rotation = 90)
data['education_num'].value_counts()
sns.countplot(data['education_num'])
data['marital_status'].value_counts()
sns.countplot(data['marital_status'])

plt.xticks(rotation = 90)
data['occupation'].value_counts()
data['occupation'] = data.occupation.str.replace('?','Unknown')
sns.countplot(data['occupation'])

plt.xticks(rotation = 90)
data['relationship'].value_counts()
sns.countplot(data['relationship'])

plt.xticks(rotation = 90)
data['race'].value_counts()
sns.countplot(data['race'])

plt.xticks(rotation = 90)
data['sex'].value_counts()
sns.countplot(data['sex'])
data['capital_loss'].value_counts().nlargest(15)
data['capital_gain'].value_counts().nlargest(15)
data['native_country'].value_counts()
data['native_country'] = data.native_country.str.replace('?','Unknown')
sns.countplot(data['native_country'])

plt.xticks(rotation = 90)
x = data.drop('income',axis = 1)

y = data['income']
x_dummy = pd.get_dummies(x)
x_train,x_test,y_train,y_test = train_test_split(x_dummy,y,test_size = 0.2,random_state = 0)
def fit_model(model,x,y):

    model.fit(x,y)

    y_pred = model.predict(x_test)

    print("Accuracy: ",model.score(x_test,y_test))

    print("------------------------------")

    print("Classification Report")

    print("------------------------------")

    print(classification_report(y_test,y_pred))

    print("------------------------------")

    print("Confusion Matrix")

    print("------------------------------")

    print(confusion_matrix(y_test,y_pred))

    print("------------------------------")
lr = LogisticRegression(max_iter = 1000)



fit_model(lr,x_train,y_train)
dtree = DecisionTreeClassifier()



fit_model(dtree,x_train,y_train)
rf = RandomForestClassifier(random_state = 0)



fit_model(rf,x_train,y_train)
gbm = GradientBoostingClassifier(random_state = 0)



fit_model(gbm,x_train,y_train)
data1 = data.copy()
data1['workclass'].value_counts() / len(data1)
names = ['State-gov','Self-emp-inc','Federal-gov','Without-pay','Never-worked']



for i in names:

    data1['workclass'] = data1.workclass.str.replace(i,'Other')
fig, ax =plt.subplots(1,2,figsize = (25,10))

sns.countplot(data['workclass'],ax = ax[0])

sns.countplot(data1['workclass'],ax = ax[1])
names1 = ['11th','9th','7th-8th','5th-6th','10th','1st-4th','Preschool','12th']



for i in names1:

    data1['education'] = data1.education.str.replace(i,'Non Graduate')

    

names2 = ['Assoc-acdm','Assoc-voc','Doctorate','Prof-school']



for i in names2:

    data1['education'] = data1.education.str.replace(i,'Other')
fig, ax =plt.subplots(1,2,figsize = (25,10))

sns.countplot(data['education'],ax = ax[0])

sns.countplot(data1['education'],ax = ax[1])
names1 = [1,2,3,4]



for i in names1:

    data1['education_num'] = data1.education_num.replace(i,'1-4')

    

names2 = [5,6,7,8]



for i in names2:

    data1['education_num'] = data1.education_num.replace(i,'5-8')

    

names3 = [9,10,11,12]



for i in names3:

    data1['education_num'] = data1.education_num.replace(i,'9-12')

    

names4 = [13,14,15,16]



for i in names4:

    data1['education_num'] = data1.education_num.replace(i,'13-16')
fig, ax =plt.subplots(1,2,figsize = (25,10))

sns.countplot(data['education_num'],ax = ax[0])

sns.countplot(data1['education_num'],ax = ax[1])
data1['marital_status'].value_counts() / len(data1)
names = ['Married-spouse-absent','Separated','Married-AF-spouse','Widowed']



for i in names:

    data1['marital_status'] = data1.marital_status.str.replace(i,'Other')
fig, ax =plt.subplots(1,2,figsize = (25,10))

sns.countplot(data['marital_status'],ax = ax[0])

sns.countplot(data1['marital_status'],ax = ax[1])
data1['occupation'].value_counts() / len(data1)
names = ['Handlers-cleaners','Transport-moving','Farming-fishing','Tech-support','Protective-serv','Armed-Forces','Priv-house-serv']



for i in names:

    data1['occupation'] = data1.occupation.str.replace(i,'Other')
fig, ax =plt.subplots(1,2,figsize = (25,10))

sns.countplot(data['occupation'],ax = ax[0])

sns.countplot(data1['occupation'],ax = ax[1])
data1['race'].value_counts() / len(data1)
names = ['Asian-Pac-Islander','Amer-Indian-Eskimo','Other']



for i in names:

    data1['race'] = data1.race.str.replace(i,'Other')
fig, ax =plt.subplots(1,2,figsize = (25,10))

sns.countplot(data['race'],ax = ax[0])

sns.countplot(data1['race'],ax = ax[1])
data1['native_country'].value_counts() / len(data1)
na = ['Cuba','Jamaica','Puerto-Rico','Honduras','Haiti','Dominican-Republic','El-Salvador','Guatemala','Nicaragua','United-States',

      'Mexico','Canada']



for i in na:

    data1['native_country'] = data1.native_country.str.replace(i,'NAmerica')

    

data1['native_country'] = data1.native_country.str.strip().replace('Outlying-US(Guam-USVI-etc)','Outlying-US')

data1['native_country'] = data1.native_country.str.replace('Outlying-US','NAmerica')



sa = ['Trinadad&Tobago','Columbia','Ecuador','Peru']



for i in sa:

    data1['native_country'] = data1.native_country.str.replace(i,'SAmerica')

    

ai = ['India','South','Iran','Philippines','Cambodia','Thailand','Laos','Taiwan','China','Japan','Vietnam','Hong']



for i in ai:

    data1['native_country'] = data1.native_country.str.replace(i,'Asia')

    

eu = ['England','Germany','Italy','Poland','Portugal','France','Yugoslavia','Scotland','Greece','Ireland','Hungary','Holand-Netherlands']



for i in eu:

    data1['native_country'] = data1.native_country.str.replace(i,'Europe')
data1.rename(columns = {'native_country':'region'}, inplace = True) 
fig, ax =plt.subplots(1,2,figsize = (25,10))

sns.countplot(data['native_country'],ax = ax[0])

sns.countplot(data1['region'],ax = ax[1])
x = data1.drop('income',axis = 1)

y = data1['income']
x_dummy = pd.get_dummies(x)
x_train,x_test,y_train,y_test = train_test_split(x_dummy,y,test_size = 0.2,random_state = 0)
lr = LogisticRegression(max_iter = 1000)



fit_model(lr,x_train,y_train)
dtree = DecisionTreeClassifier()



fit_model(dtree,x_train,y_train)
rf = RandomForestClassifier(random_state = 0)



fit_model(rf,x_train,y_train)
gbm = GradientBoostingClassifier(random_state = 0)



fit_model(gbm,x_train,y_train)
#param_grid = {'solver':['newton-cg','lblinear','lbfgs']}
#lr = LogisticRegression(max_iter = 1000)



#gs = GridSearchCV(lr,param_grid,cv = 5,scoring = 'accuracy',n_jobs = -1,verbose = True)



#gs.fit(x_train,y_train)
#gs.best_params_
#param_grid = {'penalty':['l1','l2'],

              #'C':[100.0,10.0,1.0,0.1,0.01]

    

#}
#lr = LogisticRegression(solver = 'newton-cg',penalty = 'l2',max_iter = 1000)



#gs = GridSearchCV(lr,param_grid,cv = 5,scoring = 'accuracy',n_jobs = -1,verbose = True)



#gs.fit(x_train,y_train)
#gs.best_params_
lr = LogisticRegression(C = 0.1,solver = 'newton-cg',penalty = 'l2',max_iter = 1000)



fit_model(lr,x_train,y_train)
#param_grid = {'criterion':['gini','entropy'],

              #'splitter':['best','random'],

              #'max_features':['auto','sqrt','log2'],

              #'max_depth': np.arange(2,7,1),

              #'min_samples_split': np.arange(2,10,1),

              #'min_samples_leaf': np.arange(2,7,1)

#}
#dtree = DecisionTreeClassifier()



#gs = GridSearchCV(dtree,param_grid,cv = 5,scoring = 'accuracy',n_jobs = -1,verbose = True)



#gs.fit(x_train,y_train)
#gs.best_params_
dtree = DecisionTreeClassifier(criterion = 'gini',max_depth = 6,max_features = 'auto',min_samples_leaf = 4,min_samples_split = 5,

                               splitter = 'best')



fit_model(dtree,x_train,y_train)
#param_grid = {'criterion':['gini','entropy'],

              #'bootstrap': [True,False],

              #'n_estimators':[10,100,200,500,1000],

              #'max_features':['auto','sqrt','log2'],

              #'max_depth': [2,3,4,5,6,7,None],

              #'min_samples_split': np.arange(2,10,1),

              #'min_samples_leaf': np.arange(2,7,1)

#}
#rf = RandomForestClassifier(random_state = 0)



#gs = GridSearchCV(rf,param_grid,cv = 5,scoring = 'accuracy',n_jobs = -1,verbose = True)



#gs.fit(x_train,y_train)
#gs.best_params_
rf = RandomForestClassifier(bootstrap =True,criterion = 'entropy',max_depth = None,min_samples_leaf = 2,min_samples_split = 100,

                            max_features = 17,n_estimators = 10,random_state = 0)



fit_model(rf,x_train,y_train)
#param_grid = {'n_estimators':range(20,81,10),

              #'max_depth':range(5,16,2),

              #'min_samples_split':range(1000,2100,200),

              #'min_samples_leaf':range(30,71,10),

              #'max_features':[range(7,20,2),None],

              #'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]

             #}
#gbm = GradientBoostingClassifier(n_estimators = 80,max_depth = 13,min_samples_split = 1000,min_samples_leaf = 30,max_features = None,

                                 #random_state = 0)



#gs = GridSearchCV(gbm,param_grid,cv = 5,scoring = 'accuracy',n_jobs = -1,verbose = True)



#gs.fit(x_train,y_train)
#gs.best_params_
gbm = GradientBoostingClassifier(n_estimators = 80,max_depth = 13,min_samples_split = 1000,min_samples_leaf = 30,max_features = None,

                                 random_state = 0)



fit_model(gbm,x_train,y_train)
print(plot_confusion_matrix(gbm,x_test,y_test))

print(classification_report(y_test,gbm.predict(x_test)))
plt.rcParams['figure.figsize'] = (12,8)

plt.style.use("ggplot")



gbm = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,

                           learning_rate=0.08, loss='deviance', max_depth=5,

                           max_features=None, max_leaf_nodes=None,

                           min_impurity_decrease=0.0, min_impurity_split=None,

                           min_samples_leaf=30, min_samples_split=1000,

                           min_weight_fraction_leaf=0.0, n_estimators=80,

                           n_iter_no_change=None, presort='deprecated',

                           random_state=0, subsample=1.0, tol=0.0001,

                           validation_fraction=0.1, verbose=0,

                           warm_start=False)



viz = FeatureImportances(gbm)

viz.fit(x_train, y_train)

viz.show();