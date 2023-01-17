# Import our libraries we are going to use for our data analysis.
import tensorflow as tf
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Plotly visualizations
from plotly import tools
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# For oversampling Library (Dealing with Imbalanced Datasets)
from imblearn.over_sampling import SMOTE
from collections import Counter
from IPython.display import HTML
import warnings; warnings.simplefilter('ignore')


% matplotlib inline

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
data_file = "..//input/loan_test.csv"
df = pd.read_csv(data_file,low_memory=False, index_col=0)
df.head()
df.info()
df.isnull().head()
df.isnull().sum()
df.emp_length_int.mean()
# fill in missing values with a specified value
df['emp_length_int'].fillna(value='6.05', inplace=True)
df.isnull().sum()
df.delinq_2yrs.mean()
# fill in missing values with a specified value
df['delinq_2yrs'].fillna(value='0.314', inplace=True)
df.drop('emp_length', axis=1, inplace=True)
df.isnull().sum()
df.isnull().sum()
df.drop('inq_last_6mths', axis=1, inplace=True)
df.isnull().sum()
df.annual_income.mean()
# fill in missing values with a specified value
df['annual_income'].fillna(value='75027.58', inplace=True)
df.isnull().sum()
df.open_acc.mean()
# fill in missing values with a specified value
df['open_acc'].fillna(value='11.55', inplace=True)
df.pub_rec.mean()
# fill in missing values with a specified value
df['pub_rec'].fillna(value='0.19', inplace=True)
df.revol_util.mean()
# fill in missing values with a specified value
df['revol_util'].fillna(value='55.06', inplace=True)
df.total_acc.mean()
# fill in missing values with a specified value
df['total_acc'].fillna(value='25.26', inplace=True)
df.isnull().sum()
df.collections_12_mths_ex_med.mean()
# fill in missing values with a specified value
df['collections_12_mths_ex_med'].fillna(value='0.014', inplace=True)
df.acc_now_delinq.head(10)
df.acc_now_delinq.mean()
# fill in missing values with a specified value
df['acc_now_delinq'].fillna(value='0.00', inplace=True)
df['final_d'] = pd.to_numeric(df.final_d.str.replace('/',''))
df.final_d.head(10)
df.final_d.median()
# fill in missing values with a specified value
df['final_d'].fillna(value='1012016', inplace=True)
df['next_pymnt_d'] = pd.to_numeric(df.next_pymnt_d.str.replace('/',''))
df.next_pymnt_d.median()
# fill in missing values with a specified value
df['next_pymnt_d'].fillna(value='1022016', inplace=True)
df.isnull().sum()
df['last_credit_pull_d'] = pd.to_numeric(df.last_credit_pull_d.str.replace('/',''))
df.last_credit_pull_d.median()
# fill in missing values with a specified value
df['last_credit_pull_d'].fillna(value='1012016', inplace=True)
df.isnull().sum()
df.income_category.unique()
# fill in missing values with a specified value
df['income_category'].fillna(value='Low', inplace=True)
df.isnull().sum()
# create the 'income_cat' dummy variable using the 'map' method
df['income_cat'] = df.income_category.map({'Low':1, 'Medium':2, 'High':3})
df.income_cat.mean()
df.interest_payments.unique()
# create the 'interest_payments' dummy variable using the 'map' method
df['interest_payment_cat'] = df.interest_payments.map({'Low':1, 'High':2})
df.loan_condition.unique()
# create the 'loan_condition' dummy variable using the 'map' method
df['loan_condition_cat'] = df.loan_condition.map({'Good Loan':0, 'Bad Loan':1})
df.application_type.unique()
# create the 'application_type' dummy variable using the 'map' method
df['application_type_cat'] = df.application_type.map({'INDIVIDUAL':1, 'JOINT':2})
df.purpose.isnull()
df.loan_status.unique()
# create the 'verification_status' dummy variable using the 'map' method
df['loan_status_cat'] = df.loan_status.map({'Fully Paid':1, 
                                            'Charged Off':2, 
                                            'Current':3,
                                           'Default':4, 
                                            'Late (31-120 days)':5, 
                                            'In Grace Period':6,
                                           'Late (16-30 days)':7, 
                                            'Does not meet the credit policy. Status:Fully Paid':8, 
                                            'Does not meet the credit policy. Status:Charged Off':9,
                                           'Issued':10})
df.isnull().sum()
df.verification_status.unique()
# create the 'verification_status' dummy variable using the 'map' method
df['verification_status_cat'] = df.verification_status.map({'Verified':1, 'Source Verified':2, 'Not Verified':3})
df.home_ownership.unique()
# create the 'verification_status' dummy variable using the 'map' method
df['home_ownership_cat'] = df.home_ownership.map({'RENT':1, 'OWN':2, 'MORTGAGE':3, 'OTHER':4, 'NONE':5, 'ANY':6})
df.grade.unique()
# create the 'grade' dummy variable using the 'map' method
df['grade_cat'] = df.grade.map({'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7})
df.term.unique()
# create the 'term' dummy variable using the 'map' method
df['term_cat'] = df.term.map({' 36 months':1, ' 60 months':2})
df['income_cat'] = df['income_cat'].astype(int)
df['home_ownership_cat'] = df['home_ownership_cat'].astype(int)
df['verification_status_cat'] = df['verification_status_cat'].astype(int)
df['loan_status_cat'] = df['loan_status_cat'].astype(int)
# create the 'purpose' dummy variable using the 'map' method
df['purpose_cat'] = df.purpose.map({'credit_card':1, 'car':2, 
                                            'small_business':3, 'other':4,
                                            'wedding':5, 'debt_consolidation':6,
                                            'home_improvement':7, 'major_purchase':8,
                                            'medical':9, 'moving':10,
                                            'vacation':11, 'house':12,
                                            'renewable_energy':13, 'educational':14})
df.isnull().sum()
df['application_type_cat'] = df['application_type_cat'].astype(int)
df['interest_payment_cat'] = df['interest_payment_cat'].astype(int)
df['loan_condition_cat'] = df['loan_condition_cat'].astype(int)
df['purpose_cat'] = df['purpose_cat'].astype(int)
df['grade_cat'] = df['grade_cat'].astype(int)
df['term_cat'] = df['term_cat'].astype(int)
# We have 67429 loans categorized as bad loans
badloans_df = df.loc[df["loan_condition"] == "Bad Loan"]
# loan_status cross
loan_status_cross = pd.crosstab(badloans_df['region'], badloans_df['loan_status']).apply(lambda x: x/x.sum() * 100)
number_of_loanstatus = pd.crosstab(badloans_df['region'], badloans_df['loan_status'])
number_of_loanstatus
df.loan_condition_cat.unique()
# multiple aggregation functions can be applied simultaneously
stat1=df.groupby('year').loan_amount.agg(['count', 'mean', 'min', 'max'])
df1 = pd.DataFrame(stat1)
df1
a = df1.plot(kind='bar', title='Loan statistics by year ')
# multiple aggregation functions can be applied simultaneously
stat2=df.groupby('region').loan_amount.agg(['count', 'mean', 'min', 'max'])
df2 = pd.DataFrame(stat2)
df2
b=df2.plot(kind='bar', title='Loan by Region')
df.loan_condition.head(10)
badloans_df = df.loc[df["loan_condition_cat"] == 1]
goodloans_df = df.loc[df["loan_condition_cat"] == 0]
# loan_status cross
loan_status_cross_region = pd.crosstab(badloans_df['region'], badloans_df['loan_condition_cat']).apply(lambda x: x/x.sum() * 100)
loan_status_cross_region
l = loan_status_cross_region.plot(kind='bar', title='Bad Loan percent by Region')
# loan_status cross
loan_status_cross_year = pd.crosstab(badloans_df['year'], badloans_df['loan_condition_cat']).apply(lambda x: x/x.sum() * 100)
loan_status_cross_year
m = loan_status_cross_year.plot(kind='bar', title='Bad Loan percent by Year')
loan_status=df[df.loan_condition_cat== 1].emp_length_int.value_counts()
v = loan_status.plot(kind='bar', title='Bad Loan percent by employment lenght')
loan_status=df[df.loan_condition_cat== 1].home_ownership_cat.value_counts()
a = df.home_ownership_cat.unique()
b = df.home_ownership.unique()
c = pd.DataFrame(a,b)
c
j = loan_status.plot(kind='bar', title='Bad Loan percent by Home Owner')
a = df.income_category.unique()
b = df.income_cat.unique()
c = pd.DataFrame(a,b)
c
loan_status=df[df.loan_condition_cat== 1].income_cat.value_counts()
loan_status.plot(kind='bar', title='Bad Loan percent by Income Category')
a = df.application_type.unique()
b = df.application_type_cat.unique()
c = pd.DataFrame(a,b)
c
loan_status=df[df.loan_condition_cat== 1].application_type_cat.value_counts()
t = loan_status.plot(kind='bar', title='Bad Loan percent by Application Type')
a = df.purpose.unique()
b = df.purpose_cat.unique()
c = pd.DataFrame(a,b)
c
loan_status=df[df.loan_condition_cat== 1].purpose_cat.value_counts()
loan_status.plot(kind='bar', title='Bad Loan percent by Purpose Type')
a = df.interest_payments.unique()
b = df.interest_payment_cat.unique()
c = pd.DataFrame(a,b)
c
loan_status=df[df.loan_condition_cat== 1].interest_payment_cat.value_counts()
loan_status.plot(kind='bar', title='Bad Loan percent by interest payment category Type')
a = df.grade.unique()
b = df.grade_cat.unique()
c = pd.DataFrame(a,b)
c
loan_status=df[df.loan_condition_cat== 1].grade_cat.value_counts()
loan_status.plot(kind='bar', title='Bad Loan percent by interest Grade Type')
# loan_status cross
loan_status_cross_region = pd.crosstab(badloans_df['region'], badloans_df['loan_condition_cat']).apply(lambda x: x/x.sum() * 100)
loan_status_cross_region
loan_status_cross_region.plot(kind='bar', title='Bad Loan by Region')
#df.interest_rate
# calculate the mean beer servings for each continent
stat4 = df.groupby('region').interest_rate.mean()
stat4
stat4.plot(kind='bar', x='region', y='interest rate', title='Average interest rates charged by  Banks')
stat4 = df.groupby('year').interest_rate.mean()
stat4

stat4.plot(kind='bar', x='year', y='interest rate', title='Average interest rates by year charged to customers ')
stat4 = df.groupby('year').dti.mean()
stat4
stat4.plot(kind='bar', x='year', y='debt income ratio ', title='Average debt income ratio per year charged to customers ')
stat4 = df.groupby('region').dti.mean()
stat4
stat4.plot(kind='bar', x='Region', y='debt income ratio ', title='Average debt income ratio per year charged to customers ')
df.income_cat.unique()
f, ax = plt.subplots(1,2, figsize=(16,8))

colors = ["#D72626", "#ffd733", "#42e31f"]
labels ="Low", "Medium", "High"

plt.suptitle('Information on Loan Conditions by income category', fontsize=20)

df["income_cat"].value_counts().plot.pie(explode=[0,0.25,0], autopct='%1.2f%%', ax=ax[0], shadow=True, colors=colors, 
                                             labels=labels, fontsize=12, startangle=70)

palette = ["#42e31f", "#D72626", "#ffd733"]

sns.barplot(x="year", y="income_cat", hue="loan_condition", data=df, palette=palette, estimator=lambda x: len(x) / len(df) * 100)
ax[1].set(ylabel="(%)")
# create a list of features
feature_cols = ['emp_length_int', 'annual_income','loan_amount',
                'interest_rate','dti','home_ownership_cat',
               'income_cat','total_pymnt','purpose_cat','grade_cat',
               'application_type_cat','term_cat','year']
X = df[feature_cols]
y = df.loan_condition_cat
# import class, instantiate estimator, fit with all data
from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier(n_estimators=100, max_features=3, oob_score=True, random_state=1)
rfclf.fit(df[feature_cols], df.loan_condition_cat)
# compute the feature importances
a = pd.DataFrame({'feature':feature_cols, 'importance':rfclf.feature_importances_})
model = RandomForestClassifier(n_estimators=100, max_features=3, oob_score=True, random_state=1)
model.fit(X, y)

feature_importance = model.feature_importances_
feature_importance = rfclf.feature_importances_
features = feature_cols
plt.figure(figsize=(16, 6))
plt.yscale('log', nonposy='clip')

plt.bar(range(len(feature_importance)), feature_importance, align='center')
plt.xticks(range(len(feature_importance)), features, rotation='vertical')
plt.title('Feature importance')
plt.ylabel('Importance')
plt.xlabel('Features')
plt.show()
# create a list of features
feature_cols = ['emp_length_int', 'annual_income','loan_amount',
                'interest_rate','dti','home_ownership_cat',
               'income_cat','total_pymnt','purpose_cat','grade_cat',
               'application_type_cat','term_cat','year']
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# define a function that accepts a list of features and returns testing RMSE
def train_test_rmse(feature_cols):
    X = df[feature_cols]
    y = df.loan_condition_cat
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))
# compare different sets of features
print (train_test_rmse(['emp_length_int','annual_income','loan_amount','interest_rate','dti','home_ownership_cat']))

print (train_test_rmse(['emp_length_int','annual_income','loan_amount','interest_rate','dti','home_ownership_cat','income_cat','total_pymnt','purpose_cat','grade_cat','application_type_cat','term_cat','year']))
print (train_test_rmse(['emp_length_int','annual_income','loan_amount','interest_rate','dti','home_ownership_cat','income_cat','total_pymnt','purpose_cat','grade_cat','application_type_cat','term_cat']))
print (train_test_rmse(['emp_length_int','annual_income','loan_amount','interest_rate','dti','home_ownership_cat','income_cat','total_pymnt','purpose_cat','grade_cat','application_type_cat','term_cat','year']))
# create a list of features
feature_cols = ['emp_length_int', 'annual_income','loan_amount',
                'interest_rate','dti','home_ownership_cat',
               'income_cat','total_pymnt','purpose_cat','grade_cat',
               'application_type_cat','term_cat','year']
X = df[feature_cols]
y = df.loan_condition_cat
## Modeling process
# spilt X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
Logreg = LogisticRegression()
Logreg.fit(X_train, y_train)
# make class prediction for the testing set
y_pred_class = Logreg.predict(X_test)
# calculate Accuracy
from sklearn import metrics
print((metrics.accuracy_score(y_test, y_pred_class))*100)
# examine the class distribution of the testing set (using panda series method)
y_test.value_counts()
y_test.mean()
# calculate the percentage of zeros
print ((1- y_test.mean())*100)
# claculate null accuracy ( for binary classification problem coded as 0/1)
print (max(y_test.mean(), 1- y_test.mean()))
# print the first 25 true and predicted responses
from __future__ import print_function
print('True:', y_test.values[100:250])
print('Pred:', y_pred_class[100:250])
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
# use train/test split with different random_state values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

# check classification accuracy of KNN with K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print ((metrics.accuracy_score(y_test, y_pred))*100)
# 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
from sklearn.cross_validation import cross_val_score
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print (scores)
# use average accuracy as an estimate of out-of-sample accuracy
print ((scores.mean())*100)
# search for an optimal value of K for KNN
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print (k_scores)
import matplotlib.pyplot as plt
%matplotlib inline

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
# 30-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
knn = KNeighborsClassifier(n_neighbors=30)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print (scores.mean())
# 10-fold cross-validation with the best KNN model
knn = KNeighborsClassifier(n_neighbors=2)
print ((cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())*100)
# 10-fold cross-validation with logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print ((cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())*100)
# import class, instantiate estimator, fit with all data
from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier(n_estimators=100, max_features=3, oob_score=True, random_state=1)
rfclf.fit(df[feature_cols], df.loan_condition_cat)
#compute the feature importances
a = pd.DataFrame({'feature':feature_cols, 'importance':rfclf.feature_importances_})
model = RandomForestClassifier(n_estimators=100, max_features=3, oob_score=True, random_state=1)
model.fit(X, y)

feature_importance = model.feature_importances_

# compute the out-of-bag classification accuracy
print('Mean squared error or classification error also known classification accuracy:',(rfclf.oob_score_)*100,'Percent')
# create a list of features
feature_cols = ['emp_length_int', 'annual_income','loan_amount',
                'interest_rate','dti','home_ownership_cat',
               'income_cat','total_pymnt','purpose_cat','grade_cat',
               'application_type_cat','term_cat','year']
## Modeling process
# spilt X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# import class, instantiate estimator, fit with all data
from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier(n_estimators=100, max_features=3, oob_score=True, random_state=1)
rfclf.fit(X_train, y_train)
#compute the feature importances
a = pd.DataFrame({'feature':X_train, 'importance':rfclf.feature_importances_})
model = RandomForestClassifier(n_estimators=100, max_features=3, oob_score=True, random_state=1)
model.fit(X_train, y_train)

feature_importance = model.feature_importances_

# compute the out-of-bag classification accuracy
print('Mean squared error or classification error also known classification accuracy:',(rfclf.oob_score_)*100,'Percent')