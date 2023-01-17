# import dependencies
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

sns.set(style="darkgrid")
# load data 
df = pd.read_csv("../input/HR_comma_sep.csv");

# get the column names to list
col_names = df.columns.tolist()

print("Column names:")
print(col_names)

print("\nSample data:")
df.head()
# we have 14999 rows and 10 columns
df.shape
# change the names of sales to department
df = df.rename(columns = {'sales':'department'})

df.head()
# check is the data contains 'null values'
df.isnull().any()
# check what athe departments are 
df['department'].unique()
#numpy.where(condition[, x, y])
#Return elements, either from x or y, depending on condition.

# turn support category in technical category
df['department'] = np.where(df['department'] == 'support', 'technical', df['department'])

# turn IT in technical category
df['department'] = np.where(df['department'] == 'IT' , 'technical', df['department'])

df['department'].unique()
df['left'].value_counts()
3571/11428
# check the numbers across people that left and people that didnt left

# pandas groupby function allows you to group by certain features
df.groupby('left').mean()
df.groupby('department').mean()
df.groupby('salary').mean()
# Compute a simple cross-tabulation of two (or more) factors

pd.crosstab(df.department, df.left).plot(kind='bar')
plt.title('Turnover Frequency per Department')
plt.xlabel('Department')
plt.ylabel('0; stayed | 1; left')
table = pd.crosstab(df.salary, df.left)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Turnover Frequency and Salary')
plt.xlabel('Salary')
plt.ylabel('0; stayed | 1; left')
# convert to dummies
cat_vars=['department','salary']

for var in cat_vars:
    cat_list='var'+'_'+ var
    cat_list = pd.get_dummies(df[var], prefix=var) # convert to dummy variables
    df1 = df.join(cat_list)
    df = df1
# remove the old categorical variables
df.drop(df.columns[[8,9]], axis=1, inplace=True)
df.columns.values
# the outcome variable is left (y) all the other variables are predictors

df_vars = df.columns.values.tolist()
y=['left']
X=[i for i in df_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

rfe = RFE(model, 10)
rfe = rfe.fit(df[X], df[y])
print(rfe.support_)
print('the selected features are ranked with 1')
print(rfe.ranking_)
# so these are the columns that we should select
cols = ['satisfaction_level', 'last_evaluation', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 
        'department_hr', 'department_management', 'salary_high', 'salary_low'] 
# the predictors
X = df[cols]

# the outcome 
Y = df['left']
# create a train and a test set
from sklearn.cross_validation import train_test_split

# all lowercase for random forest
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
print('Random Forest Accuracy: {:.3f}'.format(accuracy_score(y_test, rf.predict(x_test))))
import xgboost as xgb

# we first have to convert the dataset into an optimised data structure that xgb supports
data_dmatrix = xgb.DMatrix(data=X,label=Y)
from sklearn.model_selection import train_test_split

# split data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
# instantiate an XGBoost regressor object by calling the XGBregressor() class from the xgboost library
# pass the necessary hyperparameters as arguments

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(X_train,Y_train)
from sklearn.metrics import accuracy_score
print('XGBoost Accuracy: {:.3f}'.format(accuracy_score(Y_test, xg_reg.predict(X_test))))
# Random Forest model precision and recall
from sklearn.metrics import classification_report

# use sklearn to give us the report
print(classification_report(y_test, rf.predict(x_test)))
# confusion matrix for Random Forrest
y_pred = rf.predict(x_test)

from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import metrics

forest_cm = metrics.confusion_matrix(y_pred, y_test, [1,0])
sns.heatmap(forest_cm, annot=True, fmt='.2f',xticklabels = ["Left", "Stayed"] , yticklabels = ["Left", "Stayed"] )

plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Random Forest')
plt.savefig('random_forest')






