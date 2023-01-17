# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



# Support functions

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from scipy.stats import uniform



# Fit models

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



# Scoring functions

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix





import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#

x=5

y=10

z=x+y

#we don't need to explicitly convert different types of variables or set them explicitely

print('result is {}'.format(z))
#now let's see conditions

if y<x:

    print('{} is bigger than {}'.format(x,y))

elif x<y:

    print('{} is bigger than {}'.format(y,x))

else:

    print('{} is equal to {}'.format(y,x))

    

#you can play with the x and y values
#now let's define a function

def nonsense(p1,p2):

    message = ""

    if y<x:

        message = '{} is bigger than {}'.format(p1,y)

    elif x<y:

        message = '{} is bigger than {}'.format(p2,p1)

    else:

        message = '{} is equal to {}'.format(p1,p2)

    return message



#and call it

print(nonsense(x,y))
#now time to learn iterations:

#but first let's see how range function works

x = range(6)

for n in x:

  print(n)
#and time to work with arrays

my_arr = []

x = range(6)

for n in x:

  my_arr.append(n)



print (my_arr)


data = [['Alex',10],['Bob',12],['Clarke',13]]

df = pd.DataFrame(data,columns=['Name','Age'])

df
#let's introduce a new column: school

df['school']=['Primary','Secondary','Old School']

df
df[['school']]
#now let's declare a new dataframe:

df_course = pd.DataFrame([['Primary','English'],['Secondary','Physics'],['Old School','Math']])



df_course.columns=['school','course']

df_course
df1 = df.merge(df_course, on='school')

df1
df2 = pd.concat([df,df_course], axis=1)

df2
df = pd.read_csv('/kaggle/input/churn-data/Airline_Churn_Data.csv')

#df = pd.read_csv('Airline_Churn_Data.csv')

df.head()
df.info()
df.isnull().any()
df.isnull().sum()
sns.distplot(df['Age'].dropna() )
df['Age'] = df['Age'].fillna(df['Age'].median())

df.head()
df['Surname'] =  df['TierScore'].fillna('Nothing')

df['EstimatedSalary'] = df['EstimatedSalary'].fillna(df['EstimatedSalary'].median())

df['TierScore'] = df['TierScore'].fillna(df['TierScore'].median())

df['Geography'] = df['Geography'].fillna(df['Geography'].value_counts().reset_index().sort_values('Geography', ascending=False).values[0,0])

df['Gender'] = df['Gender'].fillna(df['Gender'].value_counts().reset_index().sort_values('Gender', ascending=False).values[0,0])
sns.distplot(df['MileBalance'].dropna() )
df['MileBalance'] = df['MileBalance'].fillna(df['MileBalance'].mean())
df.isnull().values.any()
df.describe()
df.nunique()
df.drop(["RowNumber"], axis = 1, inplace=True)

df.head()
#Now drop "CustomerId" and "Surname" and display df

#Your code:


fig1, ax1 = plt.subplots(figsize=(10, 8))

ax1.pie(df.Exited.value_counts(), labels=['Retained','Exited'],  autopct='%1.1f%%', startangle=90)

ax1.axis('equal')

plt.show()
corr = df.corr()



plt.figure(figsize=(18, 10))



ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True,

    annot=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
sns.pairplot(df);
fig, axarr = plt.subplots(3, 2, figsize=(20, 12))

sns.boxplot(y='TierScore',x = 'Exited', hue = 'Exited',data = df, ax=axarr[0][0])

sns.boxplot(y='Age',x = 'Exited', hue = 'Exited',data = df , ax=axarr[0][1])

sns.boxplot(y='Tenure',x = 'Exited', hue = 'Exited',data = df, ax=axarr[1][0])

sns.boxplot(y='MileBalance',x = 'Exited', hue = 'Exited',data = df, ax=axarr[1][1])

sns.boxplot(y='PrevPurchases',x = 'Exited', hue = 'Exited',data = df, ax=axarr[2][0])

sns.boxplot(y='EstimatedSalary',x = 'Exited', hue = 'Exited',data = df, ax=axarr[2][1])
fig, axarr = plt.subplots(2, 2, figsize=(20, 12))

sns.countplot(x='Geography', hue = 'Exited',data = df, ax=axarr[0][0])

sns.countplot(x='Gender', hue = 'Exited',data = df, ax=axarr[0][1])

sns.countplot(x='HasEliteCard', hue = 'Exited',data = df, ax=axarr[1][0])

sns.countplot(x='IsActiveMember', hue = 'Exited',data = df, ax=axarr[1][1])
sns.jointplot(df['MileBalance'], df['EstimatedSalary'], kind="kde", height=7, space=0)

sns.jointplot(df['MileBalance'], df['PrevPurchases'], kind="kde", height=7, space=0)

 

df['MileBalanceSalaryRatio'] = df.MileBalance/df.EstimatedSalary

df.head()
sns.jointplot(df['Age'], df['Tenure'], kind="kde", height=7, space=0)

sns.jointplot(df['Age'], df['TierScore'], kind="kde", height=7, space=0)
df['TenureByAge'] = df.Tenure/df.Age

df['TierScoreByAge'] = df.TierScore/df.Age





df.head()
sns.boxplot(y='MileBalanceSalaryRatio',x = 'Exited', hue = 'Exited',data = df)

plt.ylim(-1,5)
fig, axarr = plt.subplots(ncols=2, figsize=(20, 12))

sns.boxplot(y='TenureByAge',x = 'Exited', hue = 'Exited',data = df , ax=axarr[0])

sns.boxplot(y='TierScoreByAge',x = 'Exited', hue = 'Exited',data = df, ax=axarr[1])



#display heatmap for the correlation of new features with target

#Your code:
#let's see the distribution of each feature with each other and the target

#Your code:
cat_names = ['Gender', 'Geography', 'IsActiveMember', 'HasEliteCard']

cont_names = ['TierScore',  'Age', 'Tenure', 'MileBalance','PrevPurchases', 'EstimatedSalary', 'MileBalanceSalaryRatio', 'TenureByAge','TierScoreByAge']





df_cat = df[cat_names]

df_cont = df[cont_names]

df_target = df['Exited']

df_cat
df_cat = pd.get_dummies(df_cat)


df_all = pd.concat([df_cont,df_cat,df_target], axis = 1)

df_all.head()
x=df_all[['TierScore','Age','Tenure','MileBalance','PrevPurchases','EstimatedSalary','MileBalanceSalaryRatio','TenureByAge','TierScoreByAge','Gender_Female','Gender_Male','Geography_France','Geography_Germany','Geography_Spain', 'IsActiveMember', 'HasEliteCard']]

y=df[['Exited']]



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



print("x_train shape: {}".format(x_train.shape))

print("x_test shape: {}".format(x_test.shape))

print("y_train shape: {}".format(y_train.shape))

print("y_test shape: {}".format(y_test.shape))
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))

scaler.fit(x_train)



x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)



x_train


param_grid = {'C': [0.1,0.5,1,10,50,100], 'max_iter': [250], 'fit_intercept':[True],'intercept_scaling':[1],

              'penalty':['l2'], 'tol':[0.00001,0.0001,0.000001]}

log_primal_Grid = GridSearchCV(LogisticRegression(solver='lbfgs'),param_grid, cv=10, refit=True, verbose=0)

log_primal_Grid.fit(x_train,y_train)



print(log_primal_Grid.best_score_)    

print(log_primal_Grid.best_params_)

print(log_primal_Grid.best_estimator_)
param_grid = {'C': [0.1,10,50], 'max_iter': [300,500, 1000], 'fit_intercept':[True],'intercept_scaling':[1],'penalty':['l2'],

              'tol':[0.0001,0.000001]}

poly2 = PolynomialFeatures(degree=2)

df_train_pol2 = poly2.fit_transform(x_train)

log_pol2_Grid = GridSearchCV(LogisticRegression(solver = 'liblinear'),param_grid, cv=5, refit=True, verbose=0)

log_pol2_Grid.fit(df_train_pol2,y_train)



print(log_pol2_Grid.best_score_)    

print(log_pol2_Grid.best_params_)

print(log_pol2_Grid.best_estimator_)

# Fit random forest classifier

param_grid = {'max_depth': [3,  8], 'max_features': [2,6],'n_estimators':[50,100],'min_samples_split': [3,7]}

RanFor_grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, refit=True, verbose=0)

RanFor_grid.fit(x_train, y_train)



print(RanFor_grid.best_score_)    

print(RanFor_grid.best_params_)

print(RanFor_grid.best_estimator_)
# Fit Extreme Gradient boosting classifier

param_grid = {'max_depth': [5], 'gamma': [0.01,0.001,0.001],'min_child_weight':[1], 'learning_rate': [0.05, 0.3], 'n_estimators':[5, 100]}

xgb_grid = GridSearchCV(XGBClassifier(), param_grid, cv=5, refit=True, verbose=0)

xgb_grid.fit(x_train,y_train)



print(xgb_grid.best_score_)    

print(xgb_grid.best_params_)

print(xgb_grid.best_estimator_)
from sklearn.model_selection import learning_curve

#Ref: http://scikit-learn.org/stable/modules/learning_curve.html

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(0.05, 1.0, 20)):

    plt.figure(figsize=(10,6))

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, '-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, '-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
#change learning rate from 0.05 to 0.02 to converge better

sns.set()



kfolds = 5 #5-fold split

title = 'Learning Curves for Naive Bayes'

estimator = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=0.001,

              learning_rate=0.05, max_delta_step=0, max_depth=5,

              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,

              nthread=None, objective='binary:logistic', random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=None, subsample=1, verbosity=1)

plot_learning_curve(estimator, title, x_train, y_train, cv=kfolds)

plt.show()
poly2 = PolynomialFeatures(degree=2)

x_train_2 = poly2.fit_transform(x_train)

x_test_2 = poly2.transform(x_test)



model_lr_2 = LogisticRegression(C=50, class_weight=None, dual=False, fit_intercept=True,

                   intercept_scaling=1, l1_ratio=None, max_iter=300,

                   multi_class='warn', n_jobs=None, penalty='l2',

                   random_state=None, solver='liblinear', tol=0.0001, verbose=0,

                   warm_start=False)



model_lr_2.fit(x_train_2,y_train)

y_pred_test = model_lr_2.predict(x_test_2)



from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred_test)
#Now let's try our best model for logistic regression

#try different accuracy scores

#like f-1 score

#Your Code:
#Now let's try our best model for random forest

#try different accuracy scores

#like f-1 score

#Your Code:
#Now let's try our best model for XG Boost

#try different accuracy scores

#like f-1 score

#Your Code:
def draw_cm( actual, predicted ):

    cm = confusion_matrix( actual, predicted, [1,0] )

    sns.heatmap(cm, annot=True,  fmt='.0f', xticklabels = ["1", "0"] , 

                yticklabels = ["1", "0"] )

    plt.ylabel('ACTUAL')

    plt.xlabel('PREDICTED')

    plt.show()

draw_cm( y_test, y_pred_test)
y_base = np.ones((y_test.shape))

indices = np.random.choice(np.arange(y_base.size), replace=False, size=int(y_base.size * 0.8))

y_base[indices] = 0
draw_cm( y_test, y_base)
#now let's print our classification report for our base accuracy and any other algorithm we have trained


def get_auc_scores(y_actual, method,method2):

    auc_score = roc_auc_score(y_actual, method); 

    fpr_df, tpr_df, _ = roc_curve(y_actual, method2); 

    return (auc_score, fpr_df, tpr_df)
'''auc_xgb, fpr_xgb, tpr_xgb = get_auc_scores(y_test, y_pred_test, model_xgb.predict_proba(x_test)[:,1])

auc_rf, fpr_rf, tpr_rf = get_auc_scores(y_test, y_pred_test, model_rf.predict_proba(x_test)[:,1])

auc_lr_2, fpr_lr_2, tpr_lr_2 = get_auc_scores(y_test, y_pred_test, model_lr_2.predict_proba(poly2.transform(x_test))[:,1])

auc_lr, fpr_lr, tpr_lr = get_auc_scores(y_test, y_pred_test, model_lr.predict_proba(x_test)[:,1])





plt.figure(figsize = (12,6), linewidth= 1)



plt.plot(fpr_xgb, tpr_xgb, label = 'XGB primal Score: ' + str(round(auc_xgb, 5)))

plt.plot(fpr_rf, tpr_rf, label = 'RF primal Score: ' + str(round(auc_rf, 5)))

plt.plot(fpr_lr, tpr_lr, label = 'LR_2 primal Score: ' + str(round(auc_lr, 5)))

plt.plot(fpr_lr_2, tpr_lr_2, label = 'LR Score: ' + str(round(auc_lr_2, 5)))



plt.plot([0,1], [0,1], 'k--', label = 'Random: 0.5')

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC Curve')

plt.legend(loc='best')

#plt.savefig('roc_results_ratios.png')

plt.show()'''




