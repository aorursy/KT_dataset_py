import os
import numpy as np
import pandas as pd
import seaborn as sns
from time import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
%matplotlib inline

path="../input" #For Kaggle
#path="input"
print(os.listdir(path))
train = pd.read_csv(path+"/train.csv")
test = pd.read_csv(path+"/test.csv")
train.head()
train.info()
print('-'*50)
test.info()
# Drop PassengerId, Ticket as they are basically Row_identifier(unique ID) type columns
# Drop Cabin since it has 77%,78% missing data in train,test sets respectively making imputation infeasible
train.drop(columns=['PassengerId','Ticket','Cabin'], axis=1, inplace = True)
test.drop(columns=['Ticket','Cabin'], axis=1, inplace = True)
train.info()
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,15))
sns.countplot(x="Survived", hue="Pclass", data=train, ax=axes[0][0])
sns.countplot(x="Survived", hue="Sex", data=train, ax=axes[0][1])
sns.countplot(x="Survived", hue="SibSp", data=train, ax=axes[1][0])
sns.countplot(x="Survived", hue="Parch", data=train, ax=axes[1][1])
sns.countplot(x="Survived", hue="Embarked", data=train, ax=axes[2][0])
#1. a) Add family_size
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
#1. b) Add Title
import re
def getTitle(name):
    title = re.search('([A-Za-z]+)\.',name)
    if title:
        return title.group(1)
    return ""

train['Title'] = train['Name'].apply(getTitle)
test['Title'] = test['Name'].apply(getTitle)
pd.crosstab(train['Title'], train['Survived'])
pd.crosstab(test['Title'], test['Sex'])
#Bucket the Titles into appropriate groups
train['Title']=train['Title'].replace(['Capt','Col','Don','Dr','Jonkheer','Major','Rev','Sir','Dona'],'Rare')
train['Title']=train['Title'].replace('Ms','Miss')
train['Title']=train['Title'].replace(['Mlle','Mme','Lady','Countess'],'Mrs')
pd.crosstab(train['Title'], train['Survived'])
test['Title']=test['Title'].replace(['Capt','Col','Don','Dr','Jonkheer','Major','Rev','Sir','Dona'],'Rare')
test['Title']=test['Title'].replace('Ms','Miss')
test['Title']=test['Title'].replace(['Mlle','Mme','Lady','Countess'],'Mrs')
pd.crosstab(test['Title'], test['Sex'])
# 2a) Fare: Only test set has missing values
test['Fare'].fillna(test['Fare'].median(), inplace = True)

# 2b) Age
train['Age'].fillna(train['Age'].median(), inplace = True)
test['Age'].fillna(test['Age'].median(), inplace = True)

# 2c) Embarked
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace = True) #replace with the 1st Mode
#3.a) Bin feature Fare into groups
plt.figure(figsize=(50,200))
#fig, axes2 = plt.subplots(nrows=1, ncols=1, figsize=(50,70))
#sns.countplot(x="Fare", data=train[train['Survived']==0], ax=axes2[0])
#sns.countplot(x="Fare", data=train[train['Survived']==1], ax=axes2[0])

#sns.kdeplot(x="Fare", data=train[train['Survived']==0], ax=axes2[0])
#sns.kdeplot(train['Fare'])
#sns.countplot(train['Fare'])
sns.countplot(y="Fare", hue="Survived", data=train)
#sns.distplot(train['Fare'])
#sns.countplot(y="Fare", data=train)
#pd.crosstab(train['Fare'], train['Survived'])
train['Fare_bin'] = pd.cut(train['Fare'],bins=[0,7.125,15.1,30,60,120,1000],
                           labels=['very_low_fare', 'low_fare', 'medium_fare', 
                                   'moderate_fare', 'high_fare', 'very_high_fare'])
test['Fare_bin'] = pd.cut(train['Fare'],bins=[0,7.125,15.1,30,60,120,1000],
                           labels=['very_low_fare', 'low_fare', 'medium_fare', 
                                   'moderate_fare', 'high_fare', 'very_high_fare'])
pd.crosstab(train['Fare_bin'], train['Survived'])
#3.b) Bin Age
plt.figure(figsize=(50,100))
sns.countplot(y="Age", hue="Survived", data=train)
#0,12,20,40,60,120
train['Age_bin'] = pd.cut(train['Age'], bins=[0,12,20,40,60,120], 
                          labels=['Child','Teenage','Adult','MiddleAge','ElderAge'])
test['Age_bin'] = pd.cut(test['Age'], bins=[0,12,20,40,60,120], 
                          labels=['Child','Teenage','Adult','MiddleAge','ElderAge'])
pd.crosstab(train['Age_bin'], train['Survived'])
train = pd.get_dummies(train, columns = ["Sex","Embarked","Title","Fare_bin", "Age_bin"], prefix_sep='=', 
                             prefix=["Sex","Embarked","Title","Fare_bin", "Age_bin"])
test = pd.get_dummies(test, columns = ["Sex","Embarked","Title","Fare_bin", "Age_bin"], prefix_sep='=', 
                             prefix=["Sex","Embarked","Title","Fare_bin", "Age_bin"])
train.head()
train.drop(columns=['Name','Age','Fare'], axis=1, inplace = True)
test.drop(columns=['Name','Age','Fare'], axis=1, inplace = True)
cols = ['Pclass','SibSp','Parch','FamilySize']
train[cols] = train[cols].astype(np.uint8)
test[cols] = test[cols].astype(np.uint8)
train['Survived'] = train['Survived'].astype(np.uint8)
# Need to save this after all pre-processing done for submission
test_Pids = test['PassengerId']
test.drop(columns=['PassengerId'], axis=1, inplace = True)
train.info()
X = train.drop(columns='Survived')
Y = train['Survived']

skf = StratifiedKFold(n_splits=4,random_state=1)
for train_index, test_index in skf.split(X, Y):
    X_tr, X_val = X.iloc[train_index], X.iloc[test_index]
    Y_tr, Y_val = Y.iloc[train_index], Y.iloc[test_index]
    print('Train & Validation sets built.')
    break
#Define generalized function for Scoring current instance of model & data
'''
returns ::  a)acc: accuracy as computed on Validation set
            b)exec_time: model build/fit time
'''
def evaluate(X_tr, Y_tr, X_val, Y_val, params):
    model = LogisticRegression()
    #We should use set_params to pass parameters to model object.
    #This has the advantage over using setattr in that it allows Scikit learn to perform some validation checks on the parameters.
    model.set_params(**params)
    
    start=time()
    model.fit(X_tr,Y_tr)
    exec_time = time() - start
    
    Y_pred = model.predict(X_val)
    acc = accuracy_score(Y_val,Y_pred) * 100.0
    return acc, exec_time
C=0.001
iterations = 500
results = np.zeros((iterations, 5))

for i in range(0,iterations):    
    model_params = {'C':C,'random_state':1}
    acc_val,time_val = evaluate(X_tr, Y_tr, X_val, Y_val, model_params)
    acc_tr,time_tr = evaluate(X_tr, Y_tr, X_tr, Y_tr, model_params)
    results[i] = i+1, C, acc_tr, acc_val, time_val
    C+=0.005

res_df = pd.DataFrame(  data=results[0:,0:], 
                        index=results[0:,0],
                        columns=['Sl','C','Train_acc','Val_acc','Build_time'])
res_df['Sl'] = res_df['Sl'].astype(np.uint16)
res_df.head()
#Find value of C & Train_acc at which Valiation set acuracy is highest.
res_df[res_df['Val_acc'] == res_df['Val_acc'].max()]
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Train & Validation Set Accuracy w.r.t to Regularization parameter C')
plt.grid(True)
plt.plot(res_df['C'], res_df['Train_acc'] , 'r*-') # plotting t, a separately 
plt.plot(res_df['C'], res_df['Val_acc'] , 'b.') # plotting t, a separately
#tol=1e-6
tol=1e-10
#iterations = 50
iterations = 37
results = np.zeros((iterations, 5))

for i in range(0,iterations):    
    model_params = {'tol':tol,'random_state':1}
    acc_val,time_val = evaluate(X_tr, Y_tr, X_val, Y_val, model_params)
    acc_tr,time_tr = evaluate(X_tr, Y_tr, X_tr, Y_tr, model_params)
    results[i] = i+1, tol, acc_tr, acc_val, time_val
    #tol*=5
    tol*=2
    #print(tol)

res_df_tol = pd.DataFrame(  data=results[0:,0:], 
                        index=results[0:,0],
                        columns=['Sl','tol','Train_acc','Val_acc','Build_time'])
res_df_tol['Sl'] = res_df['Sl'].astype(np.uint16)
res_df_tol.head()
plt.figure(figsize=(20,5))
plt.xlabel('tol')
plt.ylabel('Accuracy')
plt.title('Train & Validation Set Accuracy w.r.t to Tolerance tol')
plt.grid(True)
plt.plot(res_df_tol['tol'], res_df_tol['Train_acc'] , 'r*')
plt.plot(res_df_tol['tol'], res_df_tol['Train_acc'] , 'y-')
plt.plot(res_df_tol['tol'], res_df_tol['Val_acc'] , 'b.')
#plt.plot(res_df_tol['tol'], res_df_tol['Build_time'] , 'g')
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15,10))

ax[0].set(xlabel='tol', ylabel='Accuracy')
ax[0].set_title('Train & Validation Set Accuracy w.r.t to Tolerance tol')
ax[0].grid(True)
ax[0].plot(res_df_tol['tol'], res_df_tol['Train_acc'] , 'r*')
ax[0].plot(res_df_tol['tol'], res_df_tol['Train_acc'] , 'y')
ax[0].plot(res_df_tol['tol'], res_df_tol['Val_acc'] , 'b.')

ax[1].set(xlabel='tol', ylabel='Model Build Time')
ax[1].set_title('Model Build Time w.r.t to Tolerance tol')
ax[1].grid(True)
ax[1].plot(res_df_tol['tol'], res_df_tol['Build_time'] , 'r*')
ax[1].plot(res_df_tol['tol'], res_df_tol['Build_time'] , 'y')
# 3.1 Variation of tol wrt to the Solver

tol=1e-10
iterations = 37

# There are 5 solvers. For each, we need to see their accuracy on train & validation sets plus their build time.
# Additionaly, first two columns are Sl & tol. Hence, a total of (5*3) + 2 = 17 columns reqd.
results = np.zeros((iterations, 17))
solver_list = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']

for i in range(0,iterations):    
    model_params = {'tol':tol,'random_state':1}
    results[i][0:2] = i+1, tol
    
    j = 2 #internal counter for iterating over each of the solver's results values
    for solver in solver_list:
        model_params.update({'solver': solver})
        acc_val,time_val = evaluate(X_tr, Y_tr, X_val, Y_val, model_params)
        acc_tr,time_tr = evaluate(X_tr, Y_tr, X_tr, Y_tr, model_params)
        results[i][j:j+3] = acc_tr, acc_val, time_val
        j+=3
        
    tol*=2

columns = ['Sl','tol']
for solver in solver_list:
    columns.append('Train_acc_'+solver)
    columns.append('Val_acc_'+solver)
    columns.append('Build_time_'+solver)

res_df_solver_tol = pd.DataFrame( data=results[0:,0:], 
                                  index=results[0:,0],
                                  columns=columns)
res_df_solver_tol['Sl'] = res_df_solver_tol['Sl'].astype(np.uint16)
res_df_solver_tol.head()
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15,15))

ax[0].set(xlabel='tol', ylabel='Accuracy')
ax[0].set_title('Variation in Training Data Accuracy w.r.t to Tolerance tol for different Solvers')
ax[0].grid(True)

colour_list = ['r','g','b','c','m']
for i in range(0,5):
    ax[0].plot(res_df_solver_tol['tol'],
               res_df_solver_tol['Train_acc_'+solver_list[i]],
               colour_list[i]+'.-', label=solver_list[i])
ax[0].legend()

ax[1].set(xlabel='tol', ylabel='Accuracy')
ax[1].set_title('Variation in Validation Data Accuracy w.r.t to Tolerance tol for different Solvers')
ax[1].grid(True)
for i in range(0,5):
    ax[1].plot(res_df_solver_tol['tol'],
               res_df_solver_tol['Val_acc_'+solver_list[i]] ,
               colour_list[i]+'.-', label=solver_list[i])    
ax[1].legend()
    
ax[2].set(xlabel='tol', ylabel='Build_Time')
ax[2].set_title('Variation in Model Build Time w.r.t to Tolerance tol for different Solvers')
ax[2].grid(True)
for i in range(0,5):
    ax[2].plot(res_df_solver_tol['tol'],
               res_df_solver_tol['Build_time_'+solver_list[i]] ,
               colour_list[i]+'.-', label=solver_list[i])  
ax[2].legend()

# 3.2 Variation of C wrt to the Solver

C=0.001
iterations = 500

# There are 5 solvers. For each, we need to see their accuracy on train & validation sets plus their build time.
# Additionaly, first two columns are Sl & C. Hence, a total of (5*3) + 2 = 17 columns reqd.
results = np.zeros((iterations, 17))
solver_list = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']

for i in range(0,iterations):    
    model_params = {'C':C,'random_state':1}
    results[i][0:2] = i+1, C
    
    j = 2 #internal counter for iterating over each of the solver's results values
    for solver in solver_list:
        model_params.update({'solver': solver})
        acc_val,time_val = evaluate(X_tr, Y_tr, X_val, Y_val, model_params)
        acc_tr,time_tr = evaluate(X_tr, Y_tr, X_tr, Y_tr, model_params)
        results[i][j:j+3] = acc_tr, acc_val, time_val
        j+=3
        
    C+=0.005

columns = ['Sl','C']
for solver in solver_list:
    columns.append('Train_acc_'+solver)
    columns.append('Val_acc_'+solver)
    columns.append('Build_time_'+solver)

res_df_solver_C = pd.DataFrame( data=results[0:,0:], 
                                  index=results[0:,0],
                                  columns=columns)
res_df_solver_C['Sl'] = res_df_solver_C['Sl'].astype(np.uint16)
res_df_solver_C.head()
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15,30))

ax[0].set(xlabel='C', ylabel='Accuracy')
ax[0].set_title('Variation in Training Data Accuracy w.r.t to Inverse Regularization parameter C for different Solvers')
ax[0].grid(True)

colour_list = ['r','g','b','c','m']
for i in range(0,5):
    ax[0].plot(res_df_solver_C['C'],
               res_df_solver_C['Train_acc_'+solver_list[i]],
               colour_list[i]+'-', label=solver_list[i])
ax[0].legend()

ax[1].set(xlabel='C', ylabel='Accuracy')
ax[1].set_title('Variation in Validation Data Accuracy w.r.t to Inverse Regularization parameter C for different Solvers')
ax[1].grid(True)
for i in range(0,5):
    ax[1].plot(res_df_solver_C['C'],
               res_df_solver_C['Val_acc_'+solver_list[i]] ,
               colour_list[i]+'-', label=solver_list[i])    
ax[1].legend()
    
ax[2].set(xlabel='C', ylabel='Build_Time')
ax[2].set_title('Variation in Model Build Time w.r.t to Inverse Regularization parameter C for different Solvers')
ax[2].grid(True)
for i in range(0,5):
    ax[2].plot(res_df_solver_C['C'],
               res_df_solver_C['Build_time_'+solver_list[i]] ,
               colour_list[i]+'-', label=solver_list[i])  
ax[2].legend()
max_iter=5
iterations = 40

# There are 3 solvers. For each, we need to see their accuracy on train & validation sets plus their build time.
# Additionaly, first two columns are Sl & C. Hence, a total of (3*3) + 2 = 11 columns reqd.
results = np.zeros((iterations, 11))
solver_list = ['newton-cg', 'lbfgs', 'sag']

for i in range(0,iterations):    
    model_params = {'max_iter':max_iter,'random_state':1}
    results[i][0:2] = i+1, max_iter
    
    j = 2 #internal counter for iterating over each of the solver's results values
    for solver in solver_list:
        model_params.update({'solver': solver})
        acc_val,time_val = evaluate(X_tr, Y_tr, X_val, Y_val, model_params)
        acc_tr,time_tr = evaluate(X_tr, Y_tr, X_tr, Y_tr, model_params)
        results[i][j:j+3] = acc_tr, acc_val, time_val
        j+=3
        
    max_iter += 5

columns = ['Sl','max_iter']
for solver in solver_list:
    columns.append('Train_acc_'+solver)
    columns.append('Val_acc_'+solver)
    columns.append('Build_time_'+solver)

res_df_solver_max_iter = pd.DataFrame( data=results[0:,0:], 
                                  index=results[0:,0],
                                  columns=columns)
res_df_solver_max_iter['Sl'] = res_df_solver_max_iter['Sl'].astype(np.uint16)
res_df_solver_max_iter.head()
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15,30))

ax[0].set(xlabel='max_iter', ylabel='Accuracy')
ax[0].set_title('Variation in Training Data Accuracy w.r.t to Max Iterations for different Solvers')
ax[0].grid(True)

colour_list = ['r*-','gv-','bs-']
for i in range(0,3):
    ax[0].plot(res_df_solver_max_iter['max_iter'],
               res_df_solver_max_iter['Train_acc_'+solver_list[i]],
               colour_list[i]+'-', label=solver_list[i])
ax[0].legend()

ax[1].set(xlabel='max_iter', ylabel='Accuracy')
ax[1].set_title('Variation in Validation Data Accuracy w.r.t to Max Iterations for different Solvers')
ax[1].grid(True)
for i in range(0,3):
    ax[1].plot(res_df_solver_max_iter['max_iter'],
               res_df_solver_max_iter['Val_acc_'+solver_list[i]] ,
               colour_list[i]+'-', label=solver_list[i])    
ax[1].legend()
    
ax[2].set(xlabel='max_iter', ylabel='Build_Time')
ax[2].set_title('Variation in Model Build Time w.r.t to Max Iterations for different Solvers')
ax[2].grid(True)
for i in range(0,3):
    ax[2].plot(res_df_solver_max_iter['max_iter'],
               res_df_solver_max_iter['Build_time_'+solver_list[i]] ,
               colour_list[i]+'-', label=solver_list[i])  
ax[2].legend()
# Final values for Log Reg:
model_params = {'C':0.211, 'tol':1e-6, 'solver':'liblinear', 'random_state':1}
model = LogisticRegression()
model.set_params(**model_params)
model.fit(X_tr,Y_tr)
Y_pred = model.predict(X_val)
acc = accuracy_score(Y_val,Y_pred) * 100.0
print('Final Accuracy = {}%'.format(acc))
results_logreg = model.predict(test)
submission = pd.DataFrame({
        "PassengerId": test_Pids,
        "Survived": results_logreg})
submission.to_csv("try_1_logreg.csv", index=False)
