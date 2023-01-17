# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

import statsmodels.formula.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import train_test_split

%matplotlib inline

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/HR_comma_sep.csv")
data.head(5)
data.shape
data.info()
## Let's separate numerical and categorical vaiables into 2 dfs

def sep_data(df):
    
    numerics = ['int32','float32','int64','float64']
    num_data = df.select_dtypes(include=numerics)
    cat_data = df.select_dtypes(exclude=numerics)
    
    return num_data, cat_data

num_data,cat_data = sep_data(data)
## Let's create a summary of Numerical Variables

def print_summary(x):
    
    return pd.Series([x.count(),
                        x.isnull().sum(),
                        x.mean(),
                        x.median(),
                        x.std(),
                        x.var(),
                        x.min(),
                        x.max(),
                        x.dropna().quantile(0.25),
                        x.dropna().quantile(0.75),
                        x.dropna().quantile(0.90)    
                        ],
                       index = ["Number of Observations",
                                "Missing Values",
                               "Mean",
                                "Median",
                                "Standard Deviation",
                                "Variance",
                                "Minimum Value",
                                "Maximum Value",
                                "25th Percentile",
                                "75th Percentile",
                                "90th Percentile"])

numerical_summary = num_data.apply(func = print_summary)
    
numerical_summary
## Separate X and Y variables

y = data.loc[:,'left']
X = pd.DataFrame(data.drop(columns='left'))
X.info()
plt.rcParams['figure.figsize'] = 16, 7.5

sns.heatmap(data.corr(),cmap='tab10')
## 1. Let's run VIF to check highly correlated and hence redundant variables.

features = num_data.drop(columns='left')
feature_list = "+".join(features.columns)
y, X = dmatrices('left~'+feature_list,num_data,return_type='dataframe')
vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['Features'] = X.columns
vif
## 2. Log odds plot - to see variance in data

for feature in num_data.columns.difference(['left']):
    binned = pd.cut(num_data[feature],bins=10,labels=list(range(1,11)))
    binned = binned.dropna()
    ser = num_data.groupby(binned)['left'].sum() / (num_data.groupby(binned)['left'].count() - num_data.groupby(binned)['left'].sum()) 
    ser = np.log(ser)
    fig,axes = plt.subplots(figsize=(16,8))
    sns.barplot(x=ser.index,y=ser)
    plt.ylabel('Log Odds Ratio')
## One hot encoding will be done on categorical variables - salary and department ... 
## We need to first run label coding before using OneHotEncoding

ohe_columns = data.select_dtypes(include='object').columns

le = LabelEncoder()
data_le = data.copy()

for column in ohe_columns:
    
    data_le.loc[:,column] = le.fit_transform(data_le.loc[:,column])

## One Hot Encoding method takes arrays as X, hence we need to convert features into arrays and remove headings.
X = data_le.drop(columns='left').values   ## This approach will create rows of arrays which need to be passed to OneHotEncoder
y = data_le.loc[:,'left']  ## This does not require array - hence we are just copying and not using .values

ohe = OneHotEncoder(categorical_features=[7,8]) ## This method takes index location of categorical variables in X array as input
X = ohe.fit_transform(X).toarray()

## Let's convert X into Data Frame 
X = pd.DataFrame(X)

## Maintain columns that are unaffected by OneHotEncoding separately
total_cols = data.columns
cols_maintained = total_cols.drop(['Department','salary','left'])

## Column names for OneHotEncoded Columns - One by one
## 1. For Department
for ind in range(data[ohe_columns[0]].value_counts().count()):
    
    a = X[X[ind] == 1].index.values.astype(int) ## For any column, check where is "1" present as a value after OneHotEncoding
    name_idx = a[0] ## Index of first occurance of "1"
    name = data.loc[a[0],ohe_columns[0]] ## Value in "Department" column in data DataFrame
    col_name = ohe_columns[0] + "_" + name ## Concatenate "Department_" + Value as the new column name
    X.rename(columns={ind:col_name},inplace=True) ## Rename the column

## 2. For Salary
for ind in range(data[ohe_columns[0]].value_counts().count(),(data[ohe_columns[0]].value_counts().count() + 3)):
    
    a = X[X[ind] == 1].index.values.astype(int) ## For any column, check where is "1" present as a value after OneHotEncoding
    name_idx = a[0] ## Index of first occurance of "1"
    name = data.loc[a[0],ohe_columns[1]] ## Value in "Salary" column in data DataFrame
    col_name = ohe_columns[1] + "_" + name ## Concatenate "Salary_" + Value as the new column name
    X.rename(columns={ind:col_name},inplace=True) ## Rename the column
    
## 3. For columns unchanged by OneHotEncoding
counter = 0
for ind in range((data[ohe_columns[0]].value_counts().count() + 3),(len(X.columns))):
    
    X.rename(columns={ind:cols_maintained[counter]},inplace=True)
    counter = counter + 1 
## Let's run Logistic Regression now ....
## First, we need to split data into train and test 
## Scenario 1 --> where all dummy classes are present ....

train_X, test_X, train_y, test_y = train_test_split(X,y,test_size = 0.3,random_state = 142)

model = sm.Logit(train_y,train_X)
result = model.fit()
result.summary()
result.pvalues
## Drop one of the dummy variables for each OneHotEncoded variable 

train_X_2 = train_X.drop(columns=['Department_IT','salary_high'])
## Scenario 2 --> Run Logistic on Xs with dropped data - avoiding dummy variable trap

model_2 = sm.Logit(train_y,train_X_2)
result_2 = model_2.fit()
result_2.summary()
result_2.pvalues
test_X_2 = test_X.drop(columns=['Department_IT','salary_high'])

## Create a data frame with 2 columns - one has the predicted probability and the other has the actual class 
predict_2 = pd.DataFrame(result_2.predict(test_X_2))
predict_2.rename(columns={0:'pred_prob'},inplace=True)
predict_test = pd.concat([predict_2,test_y],axis=1)
predict_test.rename(columns={'left':'actual_class'},inplace=True)
fpr_test,tpr_test,thr_test = metrics.roc_curve(test_y,predict_2)
fpr_test = pd.DataFrame(fpr_test,columns=['fpr'])
tpr_test = pd.DataFrame(tpr_test,columns=['tpr'])
thr_test = pd.DataFrame(thr_test,columns=['threshold'])
thr_df = pd.concat([fpr_test,tpr_test,thr_test],axis=1)
roc_auc = metrics.auc(fpr_test,tpr_test)

## Create a similar DataFrame for training data as well - This will be used to draw ROC
predict_train = pd.DataFrame(result_2.predict(train_X_2))
predict_train = pd.concat([predict_train,train_y],axis=1)
predict_train.rename(columns={'left':'actual_class',0:'pred_prob'},inplace=True)

## ROC for test data ....

%matplotlib inline

plt.plot(fpr_test,tpr_test,'b','AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

## Threshold for test data 
optimal_tpr = thr_df.loc[(thr_df['fpr'] < 0.5),'tpr'].max()
optimal_fpr = thr_df.loc[(thr_df['tpr'] == optimal_tpr),'fpr'].min()
optimal_thr = thr_df.loc[(thr_df['tpr'] == optimal_tpr) & (thr_df['fpr'] == optimal_fpr),'threshold']

roc_auc
## Now the above calculations show that optimal threshold for classification is 0.1238 ... Let's create a confusion matrics for this

predict_test['predicted_class'] = predict_test.apply(lambda x : 1 if x['pred_prob'] > 0.12389 else 0 ,axis=1 )
cross_tab = pd.crosstab(predict_test['actual_class'],predict_test['predicted_class'])
sns.heatmap(cross_tab, annot=True, cmap = 'Set3',fmt='.0f')

## Let's calculate accuracy, precision and recall 
cross_tab.reset_index(inplace=True)
cross_tab = pd.DataFrame(cross_tab)
cross_tab.rename(columns={0:'pred_0',1:'pred_1'},inplace=True)
overall_sum = cross_tab.sum().sum()

accuracy = ((cross_tab['pred_0'][0] + cross_tab['pred_1'][1]) / overall_sum) * 100
print("The overall accuracy of the model on Test Data is ", np.round(accuracy,1), "%")

recall = (cross_tab['pred_1'][1] / (cross_tab['pred_1'][1] + cross_tab['pred_0'][1])) * 100 
print("The Recall for model on Test Data is ", np.round(recall,1), "%")

precision = (cross_tab['pred_1'][1] / (cross_tab['pred_1'][1] + cross_tab['pred_1'][0])) * 100 
print("The Precision for model on Test Data is ", np.round(precision,1), "%")

