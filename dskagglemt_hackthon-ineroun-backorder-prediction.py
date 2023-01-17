# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read the sample file from given path.

df = pd.read_csv('/kaggle/input/Training_Dataset_v2.csv')

df.head()
df.info()
df['went_on_backorder'].unique()
df.groupby('went_on_backorder').count() 
df.went_on_backorder.value_counts(normalize=True)
df.count() 
# The last 5 rows  

df.tail()
# Some columns are not shown, so showing first 5 rows them here.

df.loc[0:4,'sales_9_month':'potential_issue']
# Some columns are not shown, so showing first 5 rows them here.

df.loc[1687856:1687860,'sales_9_month':'potential_issue']
# Summarise the non-numerical data in df

df.describe(include=['O'])
# Summarise the numerical data in df

df.describe()
# Lets get the % of each null values.

total = df.isnull().sum().sort_values(ascending=False)

percent_1 = df.isnull().sum()/df.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'], sort=False)

missing_data.head()
missing_data
# Drop the sku column

df = df.drop('sku', axis=1)
# Drop the last row

df = df[:-1]
df.tail()
# encoding categorical columns

categorical_columns = ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk','stop_auto_buy', 'rev_stop', 'went_on_backorder']



for col in categorical_columns:

    df[col] = df[col].map({'No':0, 'Yes':1})
df.info()
# Look at replacing NaNs



# Look at histogram of lead_time

df.lead_time.plot.hist()
# lead_time

df.lead_time = df.lead_time.fillna(df.lead_time.median())
# Re-check for missing values.

# Lets get the % of each null values.

total = df.isnull().sum().sort_values(ascending=False)

percent_1 = df.isnull().sum()/df.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'], sort=False)

missing_data.head()
import matplotlib.pyplot as plt  

import seaborn as sns
df.corr().round(2)
## Visiualize it on Heat Map

##Using Pearson Correlation

plt.figure(figsize=(20,20))

cor = df.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
df[df.columns[1:]].corr()['went_on_backorder'][:]
# df[['in_transit_qty', 'forecast_3_month', 'forecast_6_month']].corr()['went_on_backorder'][:]
# df.corrwith(df['went_on_backorder'])
# #Correlation with output variable

# cor_target = abs(cor["went_on_backorder"])



# #Selecting highly correlated features

# relevant_features = cor_target[cor_target>0.005]

# relevant_features
# Take a closer look at correlations with scatter plots.



# Forecast columns

forecasts = ['forecast_3_month','forecast_6_month', 'forecast_9_month']



# Pair-wise scatter plot for the forecasts

sns.pairplot(df, vars=forecasts, hue='went_on_backorder', height=3)



# Show the plot

plt.show()
# Do a pair-wise scatter plot for sales

sales = ['sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month']

sns.pairplot(df, vars=sales, hue='went_on_backorder', height=3)

plt.show()
# Similarly, lets check for in_transit_qty;min_bank along with the latest forecast and sales data, and visualize it in a pair-wise scatter plot.

# Why I select latest? As a general thumn rule.. Latest past past sales is measured.



# feature_set_1 = ['forecast_3_month', 'sales_1_month', 'in_transit_qty', 'min_bank']

# sns.pairplot(df, vars=feature_set_1, hue='went_on_backorder', height=3)

# plt.show()
# df[df['in_transit_qty'] == 'scott']
# df[df['min_bank'] == 'scott']
# Features chosen

features = ['national_inv', 'lead_time', 'sales_1_month', 'pieces_past_due', 'perf_6_month_avg',

            'local_bo_qty', 'deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop']
X = df[features]
y = df['went_on_backorder']
display(X.shape, y.shape)
from sklearn.preprocessing import MinMaxScaler 
# Use MinMaxScaler to convert features to range 0-1



scaler = MinMaxScaler()

scaler.fit(X)



X = scaler.transform(X)

X = pd.DataFrame(X, columns=features) 
X.head()
X.tail()
# split the X and y into 2 DF's aka X_train, X_valid, y_train, y_valid.

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)



print (X_train.shape, y_train.shape)

print (X_valid.shape, y_valid.shape)

# print (df.shape)
# machine learning algorithms

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



# Model Performance matrix

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, classification_report
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, y_train)
# Predict the model



Y_valid_pred_lr = logreg.predict(X_valid)
# Model Performance

print("LogisticRegression Performance --> ")

print("Score : ", round(logreg.score(X_train, y_train) * 100, 2) )



print("Accuracy Score : ", round(accuracy_score(y_valid, Y_valid_pred_lr) * 100, 2) )



print("Confusion Matrix : " )

display( confusion_matrix(y_valid, Y_valid_pred_lr) )



print("ROC AUC Score : ", roc_auc_score(y_valid, Y_valid_pred_lr) )
acc_lm = round(logreg.score(X_train, y_train) * 100, 2) 
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

Y_pred_dt = decision_tree.predict(X_valid)

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
acc_decision_tree
# Random Forest



random_forest = RandomForestClassifier(n_estimators=10)

random_forest.fit(X_train, y_train)

Y_pred_rf = random_forest.predict(X_valid)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

acc_random_forest
modelling_score = pd.DataFrame({

    'Model': ['Linear Regression','Random Forest','Decision Tree'],

    'Score': [acc_lm, acc_random_forest, acc_decision_tree]})
modelling_score.sort_values(by='Score', ascending=False)
cm_lr = confusion_matrix(y_valid, Y_valid_pred_lr)
sns.set(font_scale=1.4) # for label size

sns.heatmap(cm_lr, annot=True, annot_kws={"size": 12}) # font size



plt.show()
cm_rf = confusion_matrix(y_valid, Y_pred_rf)

cm_dt = confusion_matrix(y_valid, Y_pred_dt)
fig, (ax1, ax2, ax3) = plt.subplots(figsize=(20,5), ncols=3, nrows=1)

sns.heatmap(cm_lr, ax=ax1, annot=True, annot_kws={"size": 12})

sns.heatmap(cm_rf, ax=ax2, annot=True, annot_kws={"size": 12})

sns.heatmap(cm_dt, ax=ax3, annot=True, annot_kws={"size": 12})

plt.show()
# from sklearn.metrics import precision_recall_curve

# from sklearn.metrics import plot_precision_recall_curve
# # draw precison recall curves

# classifiers = [(decision_tree,'DecisionTreeClassifier'),

#                (random_forest,'RandomForestClassifier'),

#                (logreg,'LogisticRegression')

#                 ]

# # plt.precision_recall_curve(X_train,y_train,X_valid,y_valid,classifiers)

# disp = plot_precision_recall_curve(classifiers, X_valid, y_valid)

# disp.ax_.set_title('2-class Precision-Recall curve: '

#                    'AP={0:0.2f}'.format(average_precision))