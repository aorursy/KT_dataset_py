import warnings

import numpy as np

import pandas as pd

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

warnings.filterwarnings('ignore')



import xgboost as xgb

from scipy import stats

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import precision_score

from sklearn import preprocessing

from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data_folder = "/kaggle/input/dunnhumby-the-complete-journey/"
df = dict()

df["hh_demographic"] = pd.read_csv(data_folder + "hh_demographic.csv")

demographic=df["hh_demographic"]

demographic["MARITAL_STATUS_CODE"].replace(['A', 'B', 'U'],['Married','Unknown','Single'],inplace=True)



demographic
df["campaign_desc"] = pd.read_csv(data_folder+"campaign_desc.csv")

campaign_desc=df["campaign_desc"]

#Sort campaign by start date

campaign_desc=campaign_desc.sort_values(by=['START_DAY','CAMPAIGN'],ascending=True)

campaign_desc
#We exclude the last five campaigns filtering on campaigns starting before days 615. we don't consider campaign 20

campaign_desc = campaign_desc[campaign_desc['START_DAY']<615]
df["campaign_table"] = pd.read_csv(data_folder+"campaign_table.csv")

campaign_table=df["campaign_table"]

campaign_table.head(10)
#We call campaign the new dataframe merging the dataset

campaign = pd.merge(campaign_desc[['CAMPAIGN','START_DAY']],campaign_table[['household_key','CAMPAIGN']],on="CAMPAIGN",how="left")

#Count number of campaign per household

campaign['#campaign']=campaign.groupby(by='household_key')['CAMPAIGN'].transform('count')

#Delete useless column

campaign=campaign.drop(columns=['CAMPAIGN','START_DAY'])

#Delete duplicates

campaign.drop_duplicates(subset=['household_key', '#campaign'], keep="first", inplace=True)

campaign
#Read the coupon_redempt table

df["coupon_redempt"] = pd.read_csv(data_folder+"coupon_redempt.csv")

coupon_redempt=df["coupon_redempt"]

#Keep only coupon redeemed before DAY 615

coupon_redempt=coupon_redempt[coupon_redempt['DAY']<615]

#Drop useless columns

coupon_redempt=coupon_redempt.drop(columns=['DAY','COUPON_UPC'])

#Keep only one occurence of coupon redeemed by campaign

coupon_redempt.drop_duplicates(subset=['household_key', 'CAMPAIGN'], keep="first", inplace=True)

#Count number of campaign the customer redeemed at least one coupon

redemption_per_household=coupon_redempt.groupby(['household_key'], as_index=False)['CAMPAIGN'].agg({'redeemed': pd.Series.nunique})

redemption_per_household
#Merging of campaign and coupon redemption tables

temp = pd.merge(campaign, redemption_per_household, on=['household_key'],how="left")

#Creation of our output variable

temp["Sensitivity"]= np.where(temp["redeemed"]>0, 'Sensible', 'Not sensible')

#Creation of our aggregated dataset. We use the inner join to keep only customers for which we have the demographics data and thoose who were part of at least one campaign

dataset= pd.merge(demographic, temp[['household_key','Sensitivity']], on=['household_key'],how="inner")
# load the dataset

df["transaction_data"] = pd.read_csv(data_folder+"transaction_data.csv")

transaction=df["transaction_data"]

#Keep transaction before day 615

transaction=transaction[transaction['DAY']<615]

#Exclude transactions related to returns

transaction=transaction[transaction['SALES_VALUE']>0]

transaction=transaction[transaction['QUANTITY']>0]

transaction.head(20)
#Calculate total sales per customer

total_sales=transaction.groupby(by='household_key', as_index=False)['SALES_VALUE'].sum().rename(columns={'SALES_VALUE': 'Total_sales'})

#Calculate total number of visits per customer

total_visits=transaction.groupby(['household_key'], as_index=False)['BASKET_ID'].agg({'total_visits': pd.Series.nunique})

#Calculate median basket amount per customer

temp_basket=transaction.groupby(['household_key','BASKET_ID'], as_index=False)['SALES_VALUE'].sum()

temp_median_basket=temp_basket.groupby(['household_key'], as_index=False)['SALES_VALUE'].median().rename(columns={'SALES_VALUE': 'median_basket'})

#Calculate average product price bought per customer

temp_product=transaction.groupby(['household_key'], as_index=False)['SALES_VALUE'].mean().rename(columns={'SALES_VALUE': 'avg_price'})

dataset=dataset.merge(total_sales,on='household_key').merge(total_visits,on='household_key').merge(temp_median_basket,on='household_key').merge(temp_product,on='household_key')

dataset=dataset.drop(columns=['household_key'])

dataset
print(dataset.shape)
print(dataset.info())
dataset.describe()
categorical_vars = ['AGE_DESC','MARITAL_STATUS_CODE','INCOME_DESC','HOMEOWNER_DESC','HH_COMP_DESC','KID_CATEGORY_DESC']

num_plots = len(categorical_vars)

total_cols = 2

total_rows = num_plots//total_cols

fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols,

                        figsize=(7*total_cols, 7*total_rows), constrained_layout=True)

for i, var in enumerate(categorical_vars):

    row = i//total_cols

    pos = i % total_cols    

    plot = sns.countplot(x=var, data=dataset, ax=axs[row][pos],hue='Sensitivity',palette="Set1")
categorical_vars = ['AGE_DESC','MARITAL_STATUS_CODE','INCOME_DESC','HOMEOWNER_DESC','HH_COMP_DESC','KID_CATEGORY_DESC']

num_plots = len(categorical_vars)

total_cols = 2

total_rows = num_plots//total_cols

fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols,

                        figsize=(7*total_cols, 7*total_rows), constrained_layout=True)

for i, var in enumerate(categorical_vars):

    row = i//total_cols

    pos = i % total_cols

    plot = sns.barplot(x=var, y='median_basket',data=dataset, ax=axs[row][pos],palette="Set1",estimator=np.median)
import collections

from collections import Counter



target = dataset['Sensitivity']

counter = Counter(target)

for k,v in counter.items():

    per = v / len(target) * 100

    print('Class=%s, Count=%d, Percentage=%.2f%%' % (k, v, per))
#1. Split data into X and Y

X=dataset.drop(columns=['Sensitivity'])

Y=dataset['Sensitivity']



#2.A. Encode string class values as integers

label_encoder = preprocessing.LabelEncoder()

label_encoder = label_encoder.fit(dataset['Sensitivity'])

label_encoded_y = label_encoder.transform(dataset['Sensitivity'])



#2.A. Encode Income values as integers

X['INCOME_DESC'].replace(['Under 15K', '15-24K', '25-34K', '35-49K', '50-74K', '75-99K', '100-124K', '125-149K', '150-174K', '175-199K', '200-249K', '250K+'],[0,1,2,3,4,5,6,7,8,9,10,11],inplace=True)



#2.A. Encode Income values as integers

X['AGE_DESC'].replace(['19-24', '25-34', '35-44', '45-54', '55-64', '65+'],[0,1,2,3,4,5],inplace=True)



#2.A. Label encoding the other categorical data

labelencoder_X_1 = LabelEncoder()

X['MARITAL_STATUS_CODE'] = labelencoder_X_1.fit_transform(X['MARITAL_STATUS_CODE'])

labelencoder_X_2 = LabelEncoder()

X['HOMEOWNER_DESC'] = labelencoder_X_2.fit_transform(X['HOMEOWNER_DESC'])

labelencoder_X_3 = LabelEncoder()

X['HH_COMP_DESC'] = labelencoder_X_3.fit_transform(X['HH_COMP_DESC'])

labelencoder_X_4 = LabelEncoder()

X['HOUSEHOLD_SIZE_DESC'] = labelencoder_X_4.fit_transform(X['HOUSEHOLD_SIZE_DESC'])

X["KID_CATEGORY_DESC"].replace(['None/Unknown','3+'],[0,3],inplace=True)

X['HOUSEHOLD_SIZE_DESC'] = X.HOUSEHOLD_SIZE_DESC.astype(float)

X['KID_CATEGORY_DESC'] = X.KID_CATEGORY_DESC.astype(float)
#Let's plot the Skewness by decreasing order

num_feats=X.dtypes[X.dtypes!='object'].index

skew_feats=X[num_feats].skew().sort_values(ascending=False)

skewness=pd.DataFrame({'Skew':skew_feats})

print(skewness)
df = pd.DataFrame(data=X, columns=['AGE_DESC','MARITAL_STATUS_CODE','INCOME_DESC','HOMEOWNER_DESC','HH_COMP_DESC','KID_CATEGORY_DESC','Total_sales','total_visits','median_basket','avg_price'])

#Permet de tracer les courbes de distribution de toutes les variables

nd = pd.melt(df, value_vars =df )

n1 = sns.FacetGrid (nd, col='variable', col_wrap=5, sharex=False, sharey = False)

n1 = n1.map(sns.distplot, 'value')

n1
# Finding the relations between the variables.

plt.figure(figsize=(20,10))

c= X.corr(method='pearson')

sns.heatmap(c,annot=True)

c
#remove attribute

X=X.drop(columns=['HOUSEHOLD_SIZE_DESC'])
# Finding the relations between the variables.

plt.figure(figsize=(20,10))

c= X.corr(method='pearson')

sns.heatmap(c,annot=True)

c
X_train, X_test, y_train, y_test = train_test_split(X,label_encoded_y ,

test_size=0.3, random_state=7,shuffle=True)
#instantiate 

pt = PowerTransformer(method='yeo-johnson', standardize=True) 



#Fit the data to the powertransformer

rescaler = pt.fit(X_train)



#Lets get the Lambdas that were found

print (rescaler.lambdas_)



calc_lambdas = rescaler.lambdas_



#Transform the data 

X_train_resc = rescaler.transform(X_train)

X_test_resc=rescaler.transform(X_test)



#Pass the transformed data into a new dataframe 

df_xt = pd.DataFrame(data=X_train_resc, columns=['AGE_DESC','MARITAL_STATUS_CODE','INCOME_DESC','HOMEOWNER_DESC','HH_COMP_DESC','KID_CATEGORY_DESC','Total_sales','total_visits','median_basket','avg_price'])



df_xt.describe()
df_xt = pd.DataFrame(data=X_train_resc,columns=['AGE_DESC','MARITAL_STATUS_CODE','INCOME_DESC','HOMEOWNER_DESC','HH_COMP_DESC','KID_CATEGORY_DESC','Total_sales','total_visits','median_basket','avg_price'])

#Permet de tracer les courbes de distribution de toutes les variables

nd = pd.melt(df_xt, value_vars =df_xt )

n1 = sns.FacetGrid (nd, col='variable', col_wrap=5, sharex=False, sharey = False)

n1 = n1.map(sns.distplot, 'value')

n1
from sklearn.decomposition import PCA



pca = PCA().fit(X_train_resc)



import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12,11)



fig, ax = plt.subplots()

xi = np.arange(1, 11, step=1)

y = np.cumsum(pca.explained_variance_ratio_)



plt.ylim(0.0,1.1)

plt.plot(xi, y, marker='o', linestyle='--', color='b')



plt.xlabel('Number of Components')

plt.xticks(np.arange(0, 11, step=1)) #change from 0-based array index to 1-based human-readable label

plt.ylabel('Cumulative variance (%)')

plt.title('The number of components needed to explain variance')



plt.axhline(y=0.90, color='r', linestyle='-')

plt.text(0.5, 0.90, '90% cut-off threshold', color = 'red', fontsize=16)



ax.grid(axis='x')

plt.show()
from sklearn.decomposition import PCA

# on standardized data

pca_std = PCA(n_components=7).fit(X_train_resc)

X_train_PCA = pca_std.transform(X_train_resc)

X_test_PCA = pca_std.transform(X_test_resc)

pca_std.explained_variance_ratio_
from sklearn.model_selection import cross_val_score



#check the performance of the XGBoost model without tune parameters

# fit model on training data

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

model = xgb.XGBClassifier()

kfold = StratifiedKFold(n_splits=5, random_state=7,shuffle=True)





Accuracy = cross_val_score(model, X_train_PCA, y_train, cv=kfold,scoring='accuracy')

Precision = cross_val_score(model, X_train_PCA, y_train, cv=kfold,scoring='precision')



print("Accuracy: %.1f%% (%.1f%%)" % (Accuracy.mean()*100, Accuracy.std()*100))

print("Precision: %.1f%% (%.1f%%)" % (Precision.mean()*100, Precision.std()*100))
# grid search to tune algorithm

model = xgb.XGBClassifier()

n_estimators = [50,100, 200, 300, 400,450, 500,600,1000]

learning_rate = [0.01,0.05,0.1,0.2]

max_depth= range(2,8)

gamma=[0, 0.25, 0.5, 0.7, 0.9, 1.0]



param_grid = dict(gamma=gamma,learning_rate=learning_rate, n_estimators=n_estimators,max_depth=max_depth)

eval_set=[(X_train_PCA, y_train), (X_test_PCA, y_test)]

kfold = StratifiedKFold(n_splits=5, random_state=7,shuffle=True)

grid_search = GridSearchCV(model, param_grid,scoring='precision', n_jobs=-1, cv=kfold)

grid_result = grid_search.fit(X_train_PCA, y_train,early_stopping_rounds= 20,eval_metric= [ "logloss"],eval_set=eval_set,verbose=20)                       



# summarize result

print("Best: %.1f%% using %s" % (grid_result.best_score_*100, grid_result.best_params_))
# fit model on training data

model = xgb.XGBClassifier(learning_rate = 0.05,\

                          max_depth=2,\

                          n_estimators=200,\

                          gamma=0.7,\

                          objective = 'binary:logistic',\

                         )

fit_params={'early_stopping_rounds': 20, 

            'eval_metric': 'logloss',

            'verbose': False,

            'eval_set': [(X_train_PCA, y_train), (X_test_PCA, y_test)]}

                         

kfold =  StratifiedKFold(n_splits=5, random_state=7,shuffle=True)



Accuracy = cross_val_score(model, X_train_PCA, y_train, cv=kfold,scoring='accuracy',fit_params = fit_params)

Precision = cross_val_score(model, X_train_PCA, y_train, cv=kfold,scoring='precision',fit_params = fit_params)



print("Accuracy: %.1f%%" % (Accuracy.mean()*100))

print("Precision: %.1f%%" % (Precision.mean()*100))
from sklearn.metrics import precision_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import recall_score

from sklearn.metrics import classification_report



model = xgb.XGBClassifier(learning_rate = 0.05,\

                          max_depth=2,\

                          n_estimators=200,\

                          gamma=0.7,\

                          objective = 'binary:logistic',\

                          )



eval_set = [(X_train_PCA, y_train), (X_test_PCA, y_test)]

model.fit(X_train_PCA, y_train, early_stopping_rounds=20, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=False)

# make predictions for test data

predictions = model.predict(X_test_PCA)

# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(y_test, predictions)



roc_auc=roc_auc_score(y_test,predictions)

print("accuracy: %.2f%%" % (accuracy * 100.0))

print("Precision: %.2f%%" % (precision * 100.0))

print(classification_report(y_test, predictions,   labels=[1,0]))



# retrieve performance metrics

results = model.evals_result()

epochs = len(results['validation_0']['error'])

x_axis = range(0, epochs)

# plot log loss

fig, ax = plt.subplots()

ax.plot(x_axis, results['validation_0']['logloss'], label='Train')

ax.plot(x_axis, results['validation_1']['logloss'], label='Test')

ax.legend()

plt.ylabel('Log Loss')

plt.title('XGBoost Log Loss')

plt.show()

# plot classification error

fig, ax = plt.subplots()

ax.plot(x_axis, results['validation_0']['error'], label='Train')

ax.plot(x_axis, results['validation_1']['error'], label='Test')

ax.legend()

plt.ylabel('Classification Error')

plt.title('XGBoost Classification Error')

plt.show()
import seaborn as sb

import sklearn as sk

import matplotlib.pyplot as plt

# confusion marix for the test data

cm = sk.metrics.confusion_matrix(y_test, predictions,  labels=[1,0])



fig, ax= plt.subplots(figsize=(10,10))

sb.heatmap(cm, annot=True, fmt='g', ax = ax); 



# labels, title and ticks

ax.set_xlabel('Predicted labels');

ax.set_ylabel('True labels'); 

ax.set_title('Confusion Matrix'); 

ax.xaxis.set_ticklabels(['Sensible','Not sensible']); 

ax.yaxis.set_ticklabels(['Sensible','Not sensible']);