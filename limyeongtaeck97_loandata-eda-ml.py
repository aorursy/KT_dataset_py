# There's not much explanation.

# Sorry.


# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split 

import lightgbm as lgb

from xgboost import plot_importance

from xgboost import XGBClassifier
data = pd.read_csv('../input/loandata/Loan payments data.csv')

data.head()
data.isna().sum()
data.describe()
data.loan_status.unique()
data.Principal.unique()
data.terms.unique()
data.effective_date.unique()
data.due_date.unique()
data.education.unique()
data.Gender.unique()
plt.figure(figsize = (8, 6))



# Heatmap of correlations

sns.heatmap(data.corr(), cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)

plt.title('Correlation Heatmap');
data[['loan_status','Principal','terms']].groupby(data.education).count()
data[['loan_status','education','terms']].groupby(data.Principal).count()
grid = sns.FacetGrid(data, row='education', col='Principal', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'loan_status', 'terms', alpha=.5, ci=None)

grid.add_legend()
data[['loan_status','Principal','terms']].groupby(data.Gender).count()
age_sort = data.age.sort_values()

plt.hist(age_sort)

plt.xlabel('age')

plt.ylabel('count')

plt.show()
grid = sns.FacetGrid(data, row='Gender', col='education', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'loan_status', 'Principal', alpha=.5, ci=None)

grid.add_legend()
newdata = data.drop(['Loan_ID','due_date','paid_off_time','past_due_days'],axis=1)

newdata.head()
newdata.isna().sum()
for t,i in zip(range(len(newdata)),newdata.loan_status):

    if i == 'PAIDOFF':

        newdata.loc[t:t,'loan_status'] = 0

    elif i == 'COLLECTION':

        newdata.loc[t:t, 'loan_status'] = 1

    elif i == 'COLLECTION_PAIDOFF':

        newdata.loc[t:t, 'loan_status'] = 2
newdata.loan_status.unique()
data[['loan_status','education','terms']].groupby(data.Principal).count()
for t,i in zip(range(len(newdata)),newdata.Principal):

    if i == 300:

        newdata.loc[t:t,'Principal'] = 0

    elif i == 500:

        newdata.loc[t:t, 'Principal'] = 1

    elif i == 700:

        newdata.loc[t:t, 'Principal'] = 2

    elif i == 800:

        newdata.loc[t:t, 'Principal'] = 2

    elif i == 900:

        newdata.loc[t:t, 'Principal'] = 3

    elif i == 1000:

        newdata.loc[t:t, 'Principal'] = 3
newdata.head()
for t,i in zip(range(len(newdata)),newdata.terms):

    if i == 7:

        newdata.loc[t:t,'terms'] = 0

    elif i == 15:

        newdata.loc[t:t, 'terms'] = 1

    elif i == 30:

        newdata.loc[t:t, 'terms'] = 2
newdata.head()
# 9/8/2016 = Thursday

# 9/9/2016 = Friday

# 9/10/2016 = Saturday

...

# start Monday : 0 ~ 6

newdata.effective_date.unique()
for t,i in zip(range(len(newdata)),newdata.effective_date):

    if i == '9/8/2016':

        newdata.loc[t:t,'effective_date'] = 3

    elif i == '9/9/2016':

        newdata.loc[t:t, 'effective_date'] = 4

    elif i == '9/10/2016':

        newdata.loc[t:t, 'effective_date'] = 5

    elif i == '9/11/2016':

        newdata.loc[t:t, 'effective_date'] = 6

    elif i == '9/12/2016':

        newdata.loc[t:t, 'effective_date'] = 0

    elif i == '9/13/2016':

        newdata.loc[t:t, 'effective_date'] = 1

    elif i == '9/14/2016':

        newdata.loc[t:t, 'effective_date'] = 2

        

        
newdata.head()
for t,i in zip(range(len(newdata)),newdata.age):

    if i >= 18 and i <= 29:

        newdata.loc[t:t,'age'] = 0

    elif i >= 30 and i <= 39:

        newdata.loc[t:t, 'age'] = 1

    elif i >= 40 and i <= 49:

        newdata.loc[t:t, 'age'] = 3

    elif i >= 50:

        newdata.loc[t:t, 'age'] = 4
newdata.head()
newdata.education.unique()
for t,i in zip(range(len(newdata)),newdata.education):

    if i == 'High School or Below':

        newdata.loc[t:t,'education'] = 0

    elif i == 'college':

        newdata.loc[t:t, 'education'] = 1

    elif i == 'Bechalor':

        newdata.loc[t:t, 'education'] = 2

    elif i == 'Master or Above':

        newdata.loc[t:t, 'education'] = 3
newdata.head()
for t,i in zip(range(len(newdata)),newdata.Gender):

    if i == 'male':

        newdata.loc[t:t,'Gender'] = 0

    elif i == 'female':

        newdata.loc[t:t, 'Gender'] = 1
newdata.head()
data_x = newdata.iloc[:,1:]

data_y = newdata.loan_status
data_x.head()
data_y.head()
X_train, X_test, Y_train, Y_test = train_test_split(data_x,data_y ,test_size = 0.30, random_state = 42)
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(data_x.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
# XG boost

xgb = XGBClassifier(n_estimators=500,learning_rate=0.2, max_depth=8)

xgb.fit(X_train,Y_train)

xgb_pred = xgb.predict(X_test)

xgb_boost = round(xgb.score(X_train, Y_train) * 100, 2)

xgb_boost
gbm = lgb.LGBMClassifier(max_depth=12,

    learning_rate=1,

    n_estimators=500)

gbm.fit(X_train,Y_train)







y_pred = gbm.predict(X_test)

print('accuracy score',round(gbm.score(X_train, Y_train) * 100, 2))

lgm_classifier = round(gbm.score(X_train, Y_train) * 100, 2)
fig, ax = plt.subplots()

plot_importance(xgb, ax=ax)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree','XG boost','lightgbm classifier'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree, xgb_boost,lgm_classifier]})

models.sort_values(by='Score', ascending=False)
# Thank you