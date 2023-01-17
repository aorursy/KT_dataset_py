import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

%matplotlib inline
#This enables the lines separating the bars in the bar graphs

from matplotlib import rcParams
rcParams['patch.force_edgecolor'] = True
rcParams['patch.facecolor'] = 'b'
df_hr = pd.read_csv('../input/HR_comma_sep.csv')

df_hr.info()
df_hr.head()
df_hr.describe()
# Let's start the visualization by having a look at the correlation coefficients

plt.figure(figsize=(8,8))
sns.heatmap(data=df_hr.drop('sales', axis=1).corr(), annot=True, cmap='viridis')
plt.title('Correlation Coefficients All')
df_hr_left = df_hr[df_hr['left'] == 1] # seperate dataset for employees who left
df_hr_not_left = df_hr[df_hr['left'] == 0] # separate dataset for employees who didn't leave
#Let's have a look at the distribution of satisfaction level for different groups of employees

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(13,9))
sns.distplot(df_hr['satisfaction_level'], bins=75, ax=ax1)
ax1.set_title('Satisfaction Level Overall')

sns.kdeplot(df_hr_left['satisfaction_level'], color='red', ax=ax2, shade=True)
sns.kdeplot(df_hr_not_left['satisfaction_level'], color='green', ax=ax2, shade=True)
ax2.set_title('Satisfaction Level Left vs Not Left')
ax2.legend(['left', 'not left'])


plt.tight_layout()
sns.lmplot(x='satisfaction_level', y = 'last_evaluation', hue='left', data=df_hr, palette='viridis',
          fit_reg=False)
plt.title('Last Evaluation vs Satisfaction Level (Hue = Left)')
plt.figure(figsize=(16, 6))
sns.violinplot(x='number_project', y='satisfaction_level', data=df_hr, hue='left')
plt.title('Number of Projects vs Satisfaction Level (Hue = Left)')
sns.lmplot(x='satisfaction_level', y = 'average_montly_hours', hue='left', data=df_hr, palette='viridis',
          fit_reg=False)
plt.title('Average Monthly Hours vs Satisfaction Level (Hue = Left)')
plt.figure(figsize = (16, 6))
sns.violinplot(x='time_spend_company', y='satisfaction_level', data=df_hr, hue='left')
plt.title('Time in Company vs Satisfaction Level (Hue = Left)')
plt.figure(figsize=(12, 6))
sns.violinplot(x='salary', y='satisfaction_level', data=pd.read_csv('../input/HR_comma_sep.csv'), hue='left')
plt.title('Salary vs Satisfaction Level (Hue = Left)')
#Let's have a look at the distribution of number of projects level for different groups of employees


fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(13,9))
sns.distplot(df_hr['number_project'], ax=ax1)
ax1.set_title('Number of Projects Overall')

sns.kdeplot(df_hr_left['number_project'], color='red', ax=ax2, shade=True)
sns.kdeplot(df_hr_not_left['number_project'], color='green', ax=ax2, shade=True)
ax2.set_title('Number of Projects of Employees Left vs Not Left')
ax2.legend(['left', 'not left'])

plt.tight_layout()
plt.figure(figsize=(16, 6))
sns.violinplot('number_project', 'average_montly_hours', data=df_hr, hue='left')
plt.title('Number of Projects vs Average Monthly Hours, (Hue = Left)')
#Let's have a look at the distribution of average monthly hours for different groups of employees

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(13,9))
sns.distplot(df_hr['average_montly_hours'], bins=75, ax=ax1)
ax1.set_title('Average Monthly Hours Overall')

sns.kdeplot(df_hr_left['average_montly_hours'], color='red', ax=ax2, shade=True)
sns.kdeplot(df_hr_not_left['average_montly_hours'], color='green', ax=ax2, shade=True)
ax2.set_title('Average Monthly Hours of Employees Left vs Not Left')
ax2.legend(['left', 'not left'])


plt.tight_layout()
plt.figure(figsize=(16, 6))
sns.violinplot('time_spend_company', 'average_montly_hours', data=df_hr, hue='left')
plt.title('Time in Comapny vs Average Monthly Hours, (Hue = Left)')
plt.figure(figsize=(16, 6))
sns.violinplot('salary', 'average_montly_hours', data=pd.read_csv('../input/HR_comma_sep.csv'), hue='left')
plt.title('Salary vs Average Monthly Hours, (Hue = Left)')
#Let's have a look at the distribution of time in company for different groups of employees

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(13,9))
sns.distplot(df_hr['time_spend_company'], ax=ax1)
ax1.set_title('Time in Company of Employees Overall')

sns.kdeplot(df_hr_left['time_spend_company'], color='Red', ax=ax2, shade=True)
sns.kdeplot(df_hr_not_left['time_spend_company'], color='green', ax=ax2, shade=True)
ax2.set_title('Time in Company of Employees Left vs Not Left')
ax2.legend(['left', 'not left'])

plt.tight_layout()
#Let's have a look at the distribution of work accident for different groups of employees


fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(14,6))

sns.countplot(df_hr['Work_accident'], ax=ax1)
ax1.set_title('Work Accident of Employees Overall')

sns.countplot(df_hr_left['Work_accident'], ax=ax2)
ax2.set_title('Work Accident of Employees Left')

sns.countplot(df_hr_not_left['Work_accident'], ax=ax3)
ax3.set_title('Work Accident of Employees who did not Leave')

plt.tight_layout()

plt.figure(figsize=(6, 6))
sns.countplot(df_hr['promotion_last_5years'])
plt.title('Promotion of Employees Overall')
#Let's have a look at the distribution of departments for different groups of employees

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(13,13))

sns.countplot(df_hr['sales'], ax=ax1)
ax1.set_title('Department of Employees Overall')

sns.countplot(df_hr_left['sales'], ax=ax2)
ax2.set_title('Department of Employees who Left')

sns.countplot(df_hr_not_left['sales'], ax=ax3)
ax3.set_title('Department of Employees who did not Leave')

plt.tight_layout()
#Let's have a look at the distribution of salary for different groups of employees

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 6))
sns.countplot(pd.read_csv('../input/HR_comma_sep.csv')['salary'], ax=ax1)
ax1.set_title('Salary of Employees Overall')

sns.countplot(pd.read_csv('../input/HR_comma_sep.csv')['salary'][pd.read_csv('../input/HR_comma_sep.csv')['left'] == 1], 
              ax=ax2)
ax2.set_title('Salary of Employees who Left')

sns.countplot(pd.read_csv('../input/HR_comma_sep.csv')['salary'][pd.read_csv('../input/HR_comma_sep.csv')['left'] == 0], 
              ax=ax3)
ax3.set_title('Salary of Employees who did not Leave')
#Let's have a look at the distribution of last evalutaion for different groups of employees

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(13, 9))
sns.distplot(df_hr['last_evaluation'], bins=50, ax=ax1)
ax1.set_title('Last Evaluation of Employees Overall')

sns.kdeplot(df_hr_left['last_evaluation'], color='red', ax=ax2, shade=True)
sns.kdeplot(df_hr_not_left['last_evaluation'], color='green', ax=ax2, shade=True)
ax2.set_title('Last Evaluation of Employees Left vs Not Left')
ax2.legend(['left', 'not left'])

plt.tight_layout()
plt.figure(figsize=(16, 6))
sns.violinplot(x='number_project', y='last_evaluation', data=df_hr, hue='left')
plt.title('Number of Projects vs Last Evaluation (Hue = Left)')
sns.lmplot(x='last_evaluation', y = 'average_montly_hours', hue='left', data=df_hr, palette='viridis',
          fit_reg=False)
plt.title('Average Monthly Hours vs Satisfaction Level (Hue = Left)')
plt.figure(figsize=(16, 6))
sns.violinplot(x='time_spend_company', y='last_evaluation', data=df_hr, hue='left')
plt.title('Time in Company vs Last Evaluation (Hue = Left)')
plt.figure(figsize=(15, 6))
sns.violinplot('salary', 'last_evaluation', data=pd.read_csv('../input/HR_comma_sep.csv'), hue='left')
plt.title('Salary vs Last Evaluation, (Hue = Left)')
len(df_hr_left[df_hr_left['last_evaluation'] > df_hr['last_evaluation'].mean()])
len(df_hr_left)
#Let's have a look at the distribution of satisfaction level for highly evaluated employees.

fig = plt.figure(figsize=(16, 4))
sns.countplot(df_hr_left['satisfaction_level'][df_hr_left['last_evaluation'] > df_hr['last_evaluation'].mean()])
plt.xticks(rotation=90)
plt.title('Satisfaction Level of employees who left and had high evaluation')
#Let's have a look at the distribution of number of projects for highly evaluated employees.

fig = plt.figure(figsize=(15, 4))
sns.countplot(df_hr_left['number_project'][df_hr_left['last_evaluation'] > df_hr['last_evaluation'].mean()])
plt.title('Number of Projects of employees who left and had high evaluation')
#Let's have a look at the distribution of average monthly hours for highly evaluated employees.

fig = plt.figure(figsize=(16, 4))
sns.distplot(df_hr_left['average_montly_hours'][df_hr_left['last_evaluation'] > df_hr['last_evaluation'].mean()], bins=75)
plt.xticks(rotation=90)
plt.title('Average Monthly Hours of employees who left and had high evaluation')
#Let's have a look at the distribution of time in company for highly evaluated employees.

fig = plt.figure(figsize=(16, 5))
sns.countplot(df_hr_left['time_spend_company'][df_hr_left['last_evaluation'] > df_hr['last_evaluation'].mean()])
plt.title('Time in Company of employees who left and had high evaluation')
#Let's have a look at the distribution of salary for highly evaluated employees.

fig = plt.figure(figsize=(6, 6))
sns.countplot(df_hr['salary'][(df_hr['last_evaluation'] > df_hr['last_evaluation'].mean()) & (df_hr['left'] == 1)])
plt.title('Salary of employees who left and had high evaluation')
#Let's have a look at the distribution of promotion for highly evaluated employees.

fig = plt.figure(figsize=(6, 6))
sns.countplot(df_hr_left['promotion_last_5years'][df_hr_left['last_evaluation'] > df_hr['last_evaluation'].mean()])
plt.title('Promotion of employees who left and had high evaluation')
# mapping salaries to numerical values for machine learning purposes

map_salary = {'low': 0, 'medium': 1, 'high': 2} 
df_hr.replace({'salary': map_salary}, inplace=True)
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import scale

skb = SelectKBest(k='all') # 'all' because I need the scores and not a specific number of variables
df_hr_shuffle = df_hr.sample(frac=1, random_state=45) # shuffled as the target column had the employees 
                                                      # who left at the top and employees who didnt leave 
                                                      # after them in an order.

X = df_hr_shuffle.drop(['sales', 'left'], axis=1)
y = df_hr_shuffle['left']
X = scale(X) # it is always recommended to scale before applying to a machine learning model. One can experiment with different scales

model_skb = skb.fit(X, y)
feature_and_score = pd.DataFrame(data=model_skb.scores_, 
                                 index=df_hr_shuffle.drop(['sales', 'left'], axis=1).columns, 
                                 columns=['ANOVA F-value'])

feature_and_score = feature_and_score.sort_values(by='ANOVA F-value', axis=0, ascending=False)

feature_and_score
fig = plt.figure(figsize=(14, 6))
sns.barplot(x='ANOVA F-value', y=feature_and_score.index, data=feature_and_score)
plt.title('Importance of Varibles')
df_hr_high_eval = df_hr[df_hr['last_evaluation'] > df_hr['last_evaluation'].mean()]
df_hr_shuffle_high_eval = df_hr_high_eval.sample(frac=1, random_state=45)
X_1 = df_hr_shuffle_high_eval.drop(['sales', 'left', 'last_evaluation'], axis=1)
y_1 = df_hr_shuffle_high_eval['left']
X_1 = scale(X_1)
model_skb_1 = skb.fit(X_1, y_1)
feature_and_score_high_eval = pd.DataFrame(data=model_skb_1.scores_, 
                                           index=df_hr_shuffle.drop(['sales', 'left', 'last_evaluation'], axis=1).columns, 
                                           columns=['ANOVA F-value'])

feature_and_score_high_eval = feature_and_score_high_eval.sort_values(by='ANOVA F-value', axis=0, ascending=False)

feature_and_score_high_eval
fig = plt.figure(figsize=(14, 6))
sns.barplot(x='ANOVA F-value', y=feature_and_score_high_eval.index, data=feature_and_score_high_eval)
plt.title('Importance of Varibles for employees with High Evaluation')
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
X_lr = df_hr_shuffle.drop(['left', 'sales'], axis=1)
y_lr = df_hr_shuffle['left']
X_lr = preprocessing.scale(X_lr)
model_lr = LogisticRegression()
param_lr = {'C' : np.linspace(1, 10, 20)}
clf_lr = GridSearchCV(model_lr, param_lr, cv=5, return_train_score=False)

clf_lr.fit(X_lr, y_lr)
print('Score for Logistic Regression is: {}'.format(round(clf_lr.best_score_, 5)))
from sklearn.svm import SVC
X_svc = df_hr_shuffle.drop(['left', 'sales'], axis=1)
y_svc = df_hr_shuffle['left']
X_svc = preprocessing.scale(X_svc)
model_svc = SVC()
param_svc_1 = {'C' : np.linspace(1, 10, 10), 'kernel' : ['rbf', 'poly', 'linear', 'sigmoid']}
param_svc_2 = {'C' : np.linspace(1, 30, 30), 'kernel' : ['rbf', 'poly']}
param_svc_3 = {'C' : np.exp2(np.arange(4, 8)), 'kernel' : ['rbf'], 'gamma' : np.exp2(np.arange(2, 7))}
param_svc_4 = {'C' : np.exp2(np.arange(4, 7)), 'kernel' : ['rbf'], 'gamma' : np.exp2(np.arange(2, 4.01, 0.25))}
model_svc_final = SVC(C=16, gamma=2**3.5)
clf_svc_final = model_svc_final.fit(X_svc, y_svc)
predicted_values_svc_final = pd.Series(clf_svc_final.predict(X_svc), index=y_svc.index)
emp_might_leave = []
for index in y_svc.index:
    if (predicted_values_svc_final.loc[index] == 1) and (y_svc.loc[index] == 0):
        emp_might_leave.append(index)
print('So the employees who might leave are employees with {} indexes.'.format(emp_might_leave))
X_svc_high_eval = df_hr_shuffle_high_eval.drop(['left', 'sales'], axis=1)
y_svc_high_eval = df_hr_shuffle_high_eval['left']

X_svc_high_eval = preprocessing.scale(X_svc_high_eval)
param_svc_high_eval_1 = {'C': np.exp2(np.arange(-4, 10)), 'kernel': ['rbf'], 'gamma':np.exp2(np.arange(-15, 5))}
param_svc_high_eval_2 = {'C': np.exp2(np.arange(0, 10)), 'kernel': ['rbf'], 'gamma':np.exp2(np.arange(1, 5))}
param_svc_high_eval_3 = {'C': np.exp2(np.arange(0, 5)), 'kernel': ['rbf'], 'gamma':np.exp2(np.arange(1, 3.1, .25))}
model_svc_high_eval_final = SVC(C=2, gamma=4)
clf_svc_high_eval_final = model_svc_high_eval_final.fit(X_svc_high_eval, y_svc_high_eval)
predicted_values_svc_high_eval_final = pd.Series(clf_svc_high_eval_final.predict(X_svc_high_eval), index=y_svc_high_eval.index)
emp_might_leave_high_eval = []
for index in y_svc_high_eval.index:
    if (predicted_values_svc_high_eval_final.loc[index] == 1) and  (y_svc_high_eval.loc[index] == 0):
        emp_might_leave_high_eval.append(index)
print('So the employees with high evaluation who might leave are employees with {} indexes.'.format(emp_might_leave_high_eval))