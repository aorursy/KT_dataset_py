# Import the neccessary modules for data manipulation and visual representation

%matplotlib inline

import pandas as pd

import seaborn as sns
#Read the analytics csv file and store our dataset into a dataframe called "df"

df = pd.read_csv('/kaggle/input/hr-comma-sepcsv/HR_comma_sep.csv', index_col=None)

df
# Check to see if there are any missing values in our data set

df.isnull().any()
# Get a quick overview of what we are dealing with in our dataset

df.head()
#打印

df.head()

df['sales'].replace(['sales', 'accounting', 'hr', 'technical', 'support', 'management',

        'IT', 'product_mng', 'marketing', 'RandD'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace = True)

df['salary'].replace(['low', 'medium', 'high'], [0, 1, 2], inplace = True)
# Renaming certain columns for better readability

# Convert "department" and "salary" features to numeric types because some functions won't be able to work with string types





df = df.rename(columns={'satisfaction_level': '满意度', 

                        'last_evaluation': '评价',

                        'number_project': '工程指标',

                        'average_montly_hours': '月平均工作时间',

                        'time_spend_company': '工龄',

                        'Work_accident': '工作事故',

                        'promotion_last_5years': '升值',

                        'sales' : '部门',

                        'left' : '离职',

                        'salary':'工资'

                        })





df.head()

df
# Move the reponse variable "turnover" to the front of the table

front = df['离职']

df.drop(labels=['离职'], axis=1,inplace = True)

df.insert(0, '离职', front)

df.head()
# The dataset contains 10 columns and 14999 observations

df.shape
# Check the type of our features. 

df.dtypes
# Looks like about 76% of employees stayed and 24% of employees left. 

# NOTE: When performing cross validation, its important to maintain this turnover ratio

df = df.rename(columns={'turnover': '离职'}) 

# df = df.rename(columns={'离职': 'turnover'}) 

# turnover_rate = df.turnover.value_counts() / 14999

turnover_rate = df['离职'].sum() / 14999

turnover_rate
# Overview of summary

# On average, employees who left had a lower satisfaction level of -20%**, worked 8hours more per month, 

# had lower salary, and had a lower promotion rate

turnover_Summary = df.groupby('离职')

turnover_Summary.mean()
# Display the statistical overview of the employees

df.describe()
#Correlation Matrix

import matplotlib.pyplot as plt

df1 = df

df1=df1.rename(columns={'满意度': 'satisfaction_level', 

                        '评价': 'evaluation',

                        '工程指标': 'project_count',

                        '月平均工作时间': 'averageMonthlyHours',

                        '工龄': 'time_spend_company',

                        '工作事故': 'Work_accident',

                        '升值': 'promotion_last_5years',

                        '部门' : 'department',

                        '离职' : 'turnover',

                        '工资':'salary'

                        })



corr = df1.corr()

corr = (corr)

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

plt.title('Heatmap of Correlation Matrix')

corr
#Department   V.S.   Turnover

clarity_color_table = pd.crosstab(index=df1["department"], 

                          columns=df1["turnover"])



clarity_color_table.plot(kind="bar", 

                 figsize=(5,5),

                 stacked=True)
#Department   V.S.   Salary

clarity_color_table = pd.crosstab(index=df1["department"], 

                          columns=df1["salary"])



clarity_color_table.plot(kind="bar", 

                 figsize=(5,5),

                 stacked=True)
#Salary   V.S.   Turnover

clarity_color_table = pd.crosstab(index=df1["salary"], 

                          columns=df1["turnover"])



clarity_color_table.plot(kind="bar", 

                 figsize=(5,5),

                 stacked=True)
#projectCount V.S. turnover

clarity_color_table = pd.crosstab(index=df1["project_count"], 

                          columns=df1["turnover"])



clarity_color_table.plot(kind="bar", 

                 figsize=(5,5),

                 stacked=True)
import matplotlib.pyplot as plt 

fig = plt.figure(figsize=(10,4),)

ax=sns.kdeplot(df1.loc[(df1['turnover'] == 0),'evaluation'] , color='b',shade=True,label='no turnover')

ax=sns.kdeplot(df1.loc[(df1['turnover'] == 1),'evaluation'] , color='r',shade=True, label='turnover')

plt.title('Last evaluation')
#KDEPlot: Kernel Density Estimate Plot

fig = plt.figure(figsize=(10,4))

ax=sns.kdeplot(df1.loc[(df1['turnover'] == 0),'averageMonthlyHours'] , color='b',shade=True, label='no turnover')

ax=sns.kdeplot(df1.loc[(df1['turnover'] == 1),'averageMonthlyHours'] , color='r',shade=True, label='turnover')

plt.title('Average monthly hours worked')
#UPDATE 8/1/2017

#ProjectCount VS AverageMonthlyHours

#Looks like the average employees who stayed worked about 200hours/month. Those that had a turnover worked about 250hours/month and 150hours/month

import seaborn as sns

sns.boxplot(x="project_count", y="averageMonthlyHours", hue="turnover", data=df1)
#UPDATE 8/1/2017

#ProjectCount VS Evaluation

#Looks like employees who did not leave the company had an average evaluation of around 70% even with different projectCounts

#There is a huge skew in employees who had a turnover tho. It drastically changes after 3 projectCounts. 

#Employees that had two projects and a horrible evaluation left. Employees with more than 3 projects and super high evaluations left

import seaborn as sns

sns.boxplot(x="project_count", y="evaluation", hue="turnover", data=df1)
#Train-Test split

from sklearn.model_selection import train_test_split

label = df1.pop('turnover')

data_train, data_test, label_train, label_test = train_test_split(df1, label, test_size = 0.2, random_state = 15)
#Logistic Regression

from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()

lg.fit(data_train, label_train)

lg_score_train = lg.score(data_train, label_train)

print("Training score: ",lg_score_train)

lg_score_test = lg.score(data_test, label_test)

print("Testing score: ",lg_score_test)
#SVM

from sklearn.svm import SVC

svm = SVC()

svm.fit(data_train, label_train)

svm_score_train = svm.score(data_train, label_train)

print("Training score: ",svm_score_train)

svm_score_test = svm.score(data_test, label_test)

print("Testing score: ",svm_score_test)

#kNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(data_train, label_train)

knn_score_train = knn.score(data_train, label_train)

print("Training score: ",knn_score_train)

knn_score_test = knn.score(data_test, label_test)

print("Testing score: ",knn_score_test)
#random forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(data_train, label_train)

rfc_score_train = rfc.score(data_train, label_train)

print("Training score: ",rfc_score_train)

rfc_score_test = rfc.score(data_test, label_test)

print("Testing score: ",rfc_score_test)



y_pred_proba = rfc.predict_proba(data_test)

from sklearn.metrics import roc_curve

fpr, tpr, thres = roc_curve(label_test.values, y_pred_proba[:,1])

# 查看指标重要性

importances = rfc.feature_importances_ 

features = data_train.columns

importances_df = pd.DataFrame([features, importances], index=['Features', 'Importances']).T

importances_df.sort_values('Importances', ascending=False)
import matplotlib.pyplot as plt

plt.plot(fpr, tpr)

plt.show()
from sklearn.metrics import roc_auc_score

score = roc_auc_score(label_test.values, y_pred_proba[:,1])

score