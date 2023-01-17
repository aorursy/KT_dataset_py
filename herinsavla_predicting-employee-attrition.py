import pandas as pd
import numpy as np
test=pd.read_csv('../input/summeranalytics2020/test.csv')
train=pd.read_csv('../input/summeranalytics2020/train.csv')
train.describe().T
train.info()
train.isnull().sum()
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
plt.figure(figsize=(15,5))
sns.boxplot(y='Attrition',data=train,x='MonthlyIncome',orient='h')
train[train.MonthlyIncome>12500].Attrition.value_counts()
plt.figure(figsize=(15,5))
sns.countplot(hue='Attrition',data=train,x='Age')
plt.figure(figsize=(15,5))
sns.countplot(hue='Attrition',data=train,x='TotalWorkingYears')
plt.hist(np.sqrt(train['TotalWorkingYears']))
sns.boxplot(y='Attrition',data=train,x='YearsInCurrentRole',orient='h')
# Normalising Data
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
train[['Age','DistanceFromHome','EmployeeNumber','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','MonthlyIncome','TotalWorkingYears','YearsAtCompany']]=scaler.fit_transform(train[['Age','DistanceFromHome','EmployeeNumber','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','MonthlyIncome','TotalWorkingYears','YearsAtCompany']])
test[['Age','DistanceFromHome','EmployeeNumber','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','MonthlyIncome','TotalWorkingYears','YearsAtCompany']]=scaler.fit_transform(test[['Age','DistanceFromHome','EmployeeNumber','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','MonthlyIncome','TotalWorkingYears','YearsAtCompany']])
## Dropping Unnecessary Features based on EDA
train2=train.drop(['Id','DistanceFromHome','EmployeeNumber','Gender','NumCompaniesWorked','PercentSalaryHike','PerformanceRating','TrainingTimesLastYear','YearsWithCurrManager','Behaviour','CommunicationSkill'],axis=1)
test2=test.drop(['Id','DistanceFromHome','EmployeeNumber','Gender','NumCompaniesWorked','PercentSalaryHike','PerformanceRating','TrainingTimesLastYear','YearsWithCurrManager','Behaviour','CommunicationSkill'],axis=1)
train3=pd.get_dummies(train, drop_first=True)
X_test=pd.get_dummies(test, drop_first=True)
y_train3=pd.Series(train.Attrition)
train3.drop('Attrition', axis=1, inplace=True)
# grid searching key hyperparametres for logistic regression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# define models and parameters
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(train3, y_train3)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
C_range=[0.01,0.1,1,10,100]
solvers = ['newton-cg', 'lbfgs', 'liblinear']
max_itera=[10,100,1000,1500]

#for i in max_itera:
lr=LogisticRegression(C=0.1,random_state=0,solver='newton-cg',penalty='l2',max_iter=100)
lr.fit(train3,y_train3)
print('RUC_AUC score for Logistic Regression Trainig for  is'  +str(roc_auc_score(y_train3,lr.predict(train3))))
#print('RUC_AUC score for Logistic Regression Validation for is'   +str(roc_auc_score(y_val,lr.predict(X_val))))
print('\n')
# grid searching key hyperparameters for RandomForestClassifier

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# define models and parameters
model = RandomForestClassifier()
n_estimators = [10, 100, 1000]
max_features = ['sqrt', 'log2']
# define grid search
grid = dict(n_estimators=n_estimators,max_features=max_features)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(train3, y_train3)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
rf=RandomForestClassifier(random_state=0,max_features='log2',n_estimators=1000)
rf.fit(train3,y_train3)
print('RUC_AUC score for Random Forest Training  is'+str(roc_auc_score(y_train3,rf.predict(train3))))
# grid searching key hyperparametres for SVC

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
# define dataset

model = SVC()
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']
# define grid search
grid = dict(kernel=kernel,C=C,gamma=gamma)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(train3, y_train3)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
from sklearn.svm import SVC
C_range=[0.01,0.1,1,10,100,1000,10000]
gammas = [0.1, 1, 10, 100]
degrees = [0, 1, 2, 3, 4, 5, 6]

#for i in degrees:
svc=SVC(random_state=0,probability=True,C=50,gamma='scale',kernel='rbf')
svc.fit(train3,y_train3)
print('RUC_AUC score for SVM Training  is' +str(roc_auc_score(y_train3,svc.predict(train3))))
print('\n')
# make a prediction with a stacking ensemble

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# define the base models
level0 = list()
level0.append(('svm', SVC(random_state=0,probability=True,C=50,gamma='scale',kernel='rbf')))
level0.append(('rf', RandomForestClassifier(random_state=0,max_features='log2',n_estimators=1000)))
# define meta learner model
level1 = LogisticRegression(C=10,random_state=0,solver='liblinear',penalty='l2',max_iter=1000)
# define the stacking ensemble
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, stack_method='predict_proba')
# fit the model on all available data
model.fit(train3, y_train3)
stack_pred=model.predict_proba(X_test)
stack=[]
for i in stack_pred:
        stack.append(i[1])
stack_sub= pd.DataFrame({'ID':np.arange(1,471), 'Attrition':stack}, index=None, columns=['ID','Attrition'])
stack_sub.to_csv('stack_7.csv', index=False)