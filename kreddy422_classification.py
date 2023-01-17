import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import patsy

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

import matplotlib
import seaborn as sns

sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')


%config InlineBackend.figure_format = 'retina'
%matplotlib inline

# Load the data
#d= pd.read_csv('./WA_Fn-UseC_-Telco-Customer-Churn.csv')
d = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
#df = df1.sample(frac=0.05,random_state=123) 

d.head()





print(d.isnull().sum())
d.head()

d.index   # “the index” (aka “the labels”)

d.columns

d.shape


d.info()

d.describe()


d.dtypes



d_gen=d.groupby('gender')
d_gen.mean()
type(d_gen)
d_churn=d.groupby('Churn')
d_churn.mean()
for klass in d_churn:
    print("churn = ",klass[0])
    print(klass[1].head())
    print('----------------------------')
    print('----------------------------\\\\n')
d.groupby(['Churn','SeniorCitizen','gender','Partner']).tenure.mean()
d.pivot_table(index='Churn',values=['SeniorCitizen','gender','Partner','tenure'])
ax = sns.barplot(x="Partner", y="MonthlyCharges", hue="Churn", data=d)
d.columns
d.PaymentMethod.value_counts()
df.PaymentMethod.value_counts()
d.StreamingMovies.value_counts()


d.StreamingMovies.value_counts()
df=d
df1=d
df.PaymentMethod.head()
%%time


from sklearn.preprocessing import LabelEncoder
categorical_variables = {}

#Creating categories denoted by integers from column values
for col in df.columns:
    cat_var_name = "cat_"+ col
    cat_var_name = LabelEncoder()
    cat_var_name.fit(df[col])
    df[col] = cat_var_name.transform(df[col])
    categorical_variables[col] = cat_var_name
    

df.info()

d.head()
df.head()
df.drop('customerID',axis=1,inplace=True)
%%time
mean_corr = df.corr()

# Set the default matplotlib figure size:
fig, ax = plt.subplots(figsize=(25,25))

# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(mean_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Plot the heatmap with seaborn.
# Assign the matplotlib axis the function returns. This will let us resize the labels.
ax = sns.heatmap(mean_corr, mask=mask, ax=ax,annot=True)

# Resize the labels.
ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=19)
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=19)

# If you put plt.show() at the bottom, it prevents those useless printouts from matplotlib.
plt.show()
#Dividing our final dataset into features(explanatory variables) and labels(target variable)
X = df.loc[:, df.columns != 'Churn']
y = df.Churn
print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.25)
print('....................')
print('train shape')
print('X',X_train.shape)
print('y',y_train.shape)
print('....................')
print('....................')
print('test shape')
print('X',X_test.shape)

print('y',y_test.shape)
print('....................')

y_train.value_counts()/len(y_train)  # classification


y_test.value_counts()/len(y_train)
# LINEAR MODEL

from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression



lr_params = {
    'penalty':['l1','l2'],
    'C':np.logspace(-4, 2, 40),
    'solver':['liblinear']
}



lr_gs = GridSearchCV(LogisticRegression(), lr_params, cv=10, verbose=1)
lr_gs.fit(X_train, y_train)
best_lr = lr_gs.best_estimator_
print(lr_gs.best_params_)
print(lr_gs.best_score_)
print(best_lr)

y_pred_lr=lr_gs.predict(X_test) # predict 

from sklearn.metrics import confusion_matrix

confusion_lr = confusion_matrix(y_test,y_pred_lr)
confusion_lr
y_pp_lr = lr_gs.predict_proba(X_test)#predict probability
Y_pp_lr = pd.DataFrame(y_pp_lr, columns=['class_0_pp','class_1_pp'])
Y_pp_lr.head()
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_lr))
# calculate testing accuracy
from sklearn import metrics
print('accuracy',metrics.accuracy_score(y_test, y_pred_lr))
print('auc', metrics.roc_auc_score(y_test,  Y_pp_lr.class_1_pp))
fig, ax = plt.subplots(figsize=(12,9))

# For class 1, find the area under the curve
fpr, tpr, _ = metrics.roc_curve(y_test,  Y_pp_lr.class_1_pp)
auc = metrics.roc_auc_score(y_test,  Y_pp_lr.class_1_pp)
plt.plot(fpr, tpr, label="AUC = "+str(auc), linewidth=4, color='#9f2305')
plt.title('ROC AUC curve for binary classification', size = 20, color='r')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.legend(loc=4, fontsize = 'xx-large')
plt.show()



logreg_coefs = pd.DataFrame({'Feature':X.columns,
                             'Coef':best_lr.coef_.flatten(),
                             'Abs_coef':np.abs(best_lr.coef_.flatten())})

logreg_coefs.sort_values('Coef', inplace=True, ascending=False)
logreg_coefs
plt.figure(figsize=(30,20))
sns.set(font_scale=2.5)
sns.barplot(x='Coef',y='Feature',data=logreg_coefs.sort_values('Coef') )
plt.title(' feature coefficients')
plt.xticks(rotation=90)
plt.show()




from sklearn.ensemble import GradientBoostingClassifier
param_grid = {'n_estimators': [10,20,30,40,50],
     'max_depth': [15,20,25,30],
     'random_state': [123],
    'max_features': [2,3,4],
     'learning_rate': [0.1],
    
    }
gra=GradientBoostingClassifier()
%%time
grid_gra = GridSearchCV(estimator=gra, param_grid=param_grid, cv= 10, n_jobs=-1)
grid_gra.fit(X_train, y_train)
# examine the best model
print(grid_gra.best_score_)
print(grid_gra.best_params_)
gra_best = grid_gra.best_estimator_
y_pred_gra= grid_gra.predict(X_test)
y_pred_gra 
confusion_gra = confusion_matrix(y_test,y_pred_gra)
confusion_gra
y_pp_gra = grid_gra.predict_proba(X_test)
Y_pp_gra = pd.DataFrame(y_pp_gra, columns=['class_0_pp','class_1_pp'])
Y_pp_gra.head()
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_gra))
# calculate testing accuracy
from sklearn import metrics
print('Accuracy for GradientBoostingClassifier',metrics.accuracy_score(y_test, y_pred_gra))
print('AUC for GradientBoostingClassifier', metrics.roc_auc_score(y_test,  Y_pp_gra.class_1_pp))
fig, ax = plt.subplots(figsize=(12,9))

# For class 1, find the area under the curve
fpr, tpr, _ = metrics.roc_curve(y_test,  Y_pp_gra.class_1_pp)
auc = metrics.roc_auc_score(y_test,  Y_pp_gra.class_1_pp)
plt.plot(fpr, tpr, label="AUC = "+str(auc), linewidth=4, color='#9f2305')
plt.title('ROC AUC curve for binary classification', size = 20, color='r')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.legend(loc=4, fontsize = 'x-large')
plt.show()
grid_reg_coefs_gra = pd.DataFrame({
        'feature':X.columns,
        'importance':gra_best.feature_importances_
    })

grid_reg_coefs_gra.sort_values('importance', inplace=True, ascending=False)
grid_reg_coefs_gra
plt.figure(figsize=(30,20))
sns.set(font_scale=1.5)
sns.barplot(x='importance',y='feature',data=grid_reg_coefs_gra.sort_values('importance'))
plt.title(' feature coefficients')
plt.xticks(rotation=90)
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {'n_estimators': [10,20,30,40,50],
     'max_depth': [15,20,25,30],
     'max_features': [2,3,4],
     'random_state': [123],
     'n_jobs': [-1]
    }
model=RandomForestClassifier(random_state=42)

grid_ra = GridSearchCV(estimator=model, param_grid=param_grid, cv= 10, n_jobs=-1)
grid_ra.fit(X_train, y_train)

# examine the best model
print(grid_ra.best_score_)
print(grid_ra.best_params_)
ra_best = grid_ra.best_estimator_
ra_best 
y_pred_ra= grid_ra.predict(X_test)
y_pred_ra

confusion_ra = confusion_matrix(y_test,y_pred_ra)
confusion_ra
y_pp_ra = grid_ra.predict_proba(X_test)
Y_pp_ra = pd.DataFrame(y_pp_gra, columns=['class_0_pp','class_1_pp'])
Y_pp_ra.head()

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_ra))
# calculate testing accuracy
from sklearn import metrics
print('Accuracy for random forest',metrics.accuracy_score(y_test, y_pred_ra))
print('AUC for random forest', metrics.roc_auc_score(y_test,  Y_pp_ra.class_1_pp))
fig, ax = plt.subplots(figsize=(12,9))

# For class 1, find the area under the curve
fpr, tpr, _ = metrics.roc_curve(y_test,  Y_pp_ra.class_1_pp)
auc = metrics.roc_auc_score(y_test,  Y_pp_ra.class_1_pp)
plt.plot(fpr, tpr, label="AUC = "+str(auc), linewidth=4, color='#9f2305')
plt.title('ROC AUC curve for binary classification', size = 20, color='r')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.legend(loc=4, fontsize = 'xx-large')
plt.show()
grid_reg_coefs_ra = pd.DataFrame({
        'feature':X.columns,
        'importance':ra_best.feature_importances_
    })

grid_reg_coefs_ra.sort_values('importance', inplace=True, ascending=False)
grid_reg_coefs_ra

plt.figure(figsize=(30,20))
sns.set(font_scale=1.5)
sns.barplot(x='importance',y='feature',data=grid_reg_coefs_ra.sort_values('importance'))
plt.title(' feature coefficients')
plt.xticks(rotation=90)
plt.show()

Acc=pd.DataFrame({
    
        
        'gradient boosting':[metrics.accuracy_score(y_test, y_pred_gra)],
        'logistic':[metrics.accuracy_score(y_test, y_pred_lr)],
        'random forest':[metrics.accuracy_score(y_test, y_pred_ra)],
    
    }, index=['Accuracy'])





from sklearn.metrics import precision_score
prec= pd.DataFrame({
        
        'gradient boosting':[precision_score(y_test, y_pred_gra)],
        'logistic':[precision_score(y_test, y_pred_lr)],
        'random forest':[precision_score(y_test, y_pred_ra)],
    
    }, index=['precision'])




from sklearn.metrics import precision_score
prec= pd.DataFrame({
        
        'gradient boosting':[precision_score(y_test, y_pred_gra)],
        'logistic':[precision_score(y_test, y_pred_lr)],
        'random forest':[precision_score(y_test, y_pred_ra)],
    
    }, index=['precision'])


from sklearn.metrics import recall_score

recall= pd.DataFrame({
        
        'gradient boosting':[recall_score(y_test, y_pred_gra)],
        'logistic':[recall_score(y_test, y_pred_lr)],
        'random forest':[recall_score(y_test, y_pred_ra)],
    
    }, index=['recall'])
AUC= pd.DataFrame({
    
        
        'gradient boosting':[metrics.roc_auc_score(y_test, Y_pp_gra.class_1_pp)],
        'logistic':[metrics.roc_auc_score(y_test, Y_pp_lr.class_1_pp)],
        'random forest':[metrics.roc_auc_score(y_test, Y_pp_ra.class_1_pp)],
    
    }, index=['roc_auc'])

acc = pd.concat([Acc, AUC,prec,recall], axis=0)
acc
f=pd.DataFrame({
        'Feature':X.columns,
        'gradient boosting':gra_best.feature_importances_,
        'decision tree':best_lr.coef_.flatten(),
        'random forest':ra_best.feature_importances_,
    
    })
f.sort_values('gradient boosting', inplace=True, ascending=False)
f.head()
f.tail()
df.pivot_table(index='Churn',values=['tenure','MonthlyCharges','TotalCharges'])
f.plot(x = "Feature", kind='barh', figsize=(11,11))




