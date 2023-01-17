import gc

gc.collect()
import pandas as pd 

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime 

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import classification_report,confusion_matrix

from sklearn import metrics

import xgboost as xgb

from xgboost.sklearn import XGBClassifier  

from xgboost.sklearn import XGBRegressor



from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer



from scipy.special import boxcox1p, inv_boxcox1p

from scipy.stats import skew,norm



import xgboost as xgb

from xgboost import XGBClassifier

from xgboost import plot_importance



import warnings

warnings.filterwarnings('ignore')
def countOutlier(df_in, col_name):

    if df_in[col_name].nunique() > 2:

        orglength = len(df_in[col_name])

        q1 = df_in[col_name].quantile(0.00)

        q3 = df_in[col_name].quantile(0.95)

        iqr = q3-q1 #Interquartile range 

        fence_low  = q1-1.5*iqr 

        fence_high = q3+1.5*iqr 

        df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]

        newlength = len(df_out[col_name])

        return round(100 - (newlength*100/orglength),2)  

    else:

        return 0



def drop_columns(dataframe, axis =1, percent=0.3):

    '''

    * drop_columns function will remove the rows and columns based on parameters provided.

    * dataframe : Name of the dataframe  

    * axis      : axis = 0 defines drop rows, axis =1(default) defines drop columns    

    * percent   : percent of data where column/rows values are null,default is 0.3(30%)

    '''

    df = dataframe.copy()

    ishape = df.shape

    if axis == 0:

        rownames = df.transpose().isnull().sum()

        rownames = list(rownames[rownames.values > percent*len(df)].index)

        df.drop(df.index[rownames],inplace=True) 

        print("\nNumber of Rows dropped\t: ",len(rownames))

    else:

        colnames = (df.isnull().sum()/len(df))

        colnames = list(colnames[colnames.values>=percent].index)

        df.drop(labels = colnames,axis =1,inplace=True)        

        print("Number of Columns dropped\t: ",len(colnames))

        

    print("\nOld dataset rows,columns",ishape,"\nNew dataset rows,columns",df.shape)



    return df



def correlation(df, dftest, threshold):

    col_corr = set() # Set of all the names of deleted columns

    corr_matrix = df.corr()

    for i in range(len(corr_matrix.columns)):

        for j in range(i):

            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):

                colname = corr_matrix.columns[i] # getting the name of column

                col_corr.add(colname)

                if colname in df.columns:

                    del df[colname] # deleting the column from the train dataset

                    del dftest[colname] # deleting the column from the test dataset



    print(df.shape)

    print(dftest.shape)
df_all = pd.read_csv('../input/prudential-life-insurance-assessment/train.csv')

test_all = pd.read_csv('../input/prudential-life-insurance-assessment/test.csv')
df_all.head()
df_all.shape
df_all.columns
df_all['Response'].value_counts()
df_all.describe()
df_all.dtypes
df_all.shape
#Checking null values in each cols.

(df_all.isnull().sum()*100/df_all.shape[0]).sort_values(ascending=False)
#Checking null values in each cols.

(test_all.isnull().sum()*100/df_all.shape[0]).sort_values(ascending=False)
# Plotting null values for first 13 columns

isna_train = df_all.isnull().sum().sort_values(ascending=False)

isna_train[:13].plot(kind='bar')
# Plotting null values for first 13 columns

isna_train = test_all.isnull().sum().sort_values(ascending=False)

isna_train[:13].plot(kind='bar')
df_all = drop_columns(df_all, axis =1, percent=0.3)

test_all = drop_columns(test_all, axis =1, percent=0.3)
#Checking null values in each cols.

(df_all.isnull().sum()*100/df_all.shape[0]).sort_values(ascending=False)
#Checking null values in each cols.

(test_all.isnull().sum()*100/test_all.shape[0]).sort_values(ascending=False)
df_all.Medical_History_1.value_counts()
df_all.Employment_Info_4.value_counts()
df_all.Employment_Info_6.value_counts()
df_all.Employment_Info_1.value_counts()
df_all.Employment_Info_1.fillna(0, inplace=True)

df_all.Employment_Info_6.fillna(1, inplace=True)

df_all.Employment_Info_4.fillna(0, inplace=True)

df_all.Medical_History_1.fillna(1, inplace=True)



test_all.Employment_Info_1.fillna(0, inplace=True)

test_all.Employment_Info_6.fillna(1, inplace=True)

test_all.Employment_Info_4.fillna(0, inplace=True)

test_all.Medical_History_1.fillna(1, inplace=True)
#Checking null values in each cols.

(df_all.isnull().sum()*100/df_all.shape[0]).sort_values(ascending=False)
#Checking null values in each cols.

(test_all.isnull().sum()*100/test_all.shape[0]).sort_values(ascending=False)
df_all.shape
test_all.shape
print(df_all.isnull().values.any())

print(test_all.isnull().values.any())
sns.set_color_codes()

plt.figure(figsize=(8,8))

sns.countplot(df_all.Response).set_title('Dist of Response variables')
f, axes = plt.subplots(1, 2, figsize=(10,5))

sns.boxplot(x = 'BMI', data=df_all,  orient='v' , ax=axes[0])

sns.distplot(df_all['BMI'],  ax=axes[1])
countOutlier(df_all,'BMI')
f, axes = plt.subplots(1, 2, figsize=(10,5))

sns.boxplot(x = 'Ins_Age', data=df_all,  orient='v' , ax=axes[0])

sns.distplot(df_all['Ins_Age'],  ax=axes[1])
countOutlier(df_all,'Ins_Age')
f, axes = plt.subplots(1, 2, figsize=(10,5))

sns.boxplot(x = 'Ht', data=df_all,  orient='v' , ax=axes[0])

sns.distplot(df_all['Ht'],  ax=axes[1])
countOutlier(df_all,'Ht')
f, axes = plt.subplots(1, 2, figsize=(10,5))

sns.boxplot(x = 'Wt', data=df_all,  orient='v' , ax=axes[0])

sns.distplot(df_all['Wt'],  ax=axes[1])
countOutlier(df_all,'Wt')
#1 Multiplication of BMI and Age - higher the factor, higher the risk

df_all['bmi_age'] = df_all['BMI'] * df_all['Ins_Age']



#2 Multiplication of Weight and Age - higher the factor, higher the risk

df_all['age_wt'] = df_all['Ins_Age'] * df_all['Wt']



#3 Multiplication of Height and Age - higher the factor, higher the risk

df_all['age_ht'] = df_all['Ins_Age'] * df_all['Ht']



#4 Add all values of Medical Keywords Columns

med_keyword_columns = df_all.columns[df_all.columns.str.startswith('Medical_Keyword_')]

df_all['med_keywords_count'] = df_all[med_keyword_columns].sum(axis=1)

df_all.drop(med_keyword_columns, axis=1, inplace=True)



#5 BMI Categorization

conditions = [

    (df_all['BMI'] <= df_all['BMI'].quantile(0.25)),

    (df_all['BMI'] > df_all['BMI'].quantile(0.25)) & (df_all['BMI'] <= df_all['BMI'].quantile(0.75)),

    (df_all['BMI'] > df_all['BMI'].quantile(0.75))]

choices = ['under_weight', 'average', 'overweight']

df_all['bmi_wt'] = np.select(conditions, choices)



#6 Age Categorization

conditions = [

    (df_all['Ins_Age'] <= df_all['Ins_Age'].quantile(0.25)),

    (df_all['Ins_Age'] > df_all['Ins_Age'].quantile(0.25)) & (df_all['Ins_Age'] <= df_all['Ins_Age'].quantile(0.75)),

    (df_all['Ins_Age'] > df_all['Ins_Age'].quantile(0.75))]

choices = ['young', 'average', 'old']

df_all['age_cat'] = np.select(conditions, choices)



#7 Height Categorization

conditions = [

    (df_all['Ht'] <= df_all['Ht'].quantile(0.25)),

    (df_all['Ht'] > df_all['Ht'].quantile(0.25)) & (df_all['Ht'] <= df_all['Ht'].quantile(0.75)),

    (df_all['Ht'] > df_all['Ht'].quantile(0.75))]

choices = ['short', 'average', 'tall']

df_all['ht_cat'] = np.select(conditions, choices)



#8 Weight Categorization

conditions = [

    (df_all['Wt'] <= df_all['Wt'].quantile(0.25)),

    (df_all['Wt'] > df_all['Wt'].quantile(0.25)) & (df_all['Wt'] <= df_all['Wt'].quantile(0.75)),

    (df_all['Wt'] > df_all['Wt'].quantile(0.75))]

choices = ['thin', 'average', 'fat']

df_all['wt_cat'] = np.select(conditions, choices)
#1 Multiplication of BMI and Age - higher the factor, higher the risk

test_all['bmi_age'] = test_all['BMI'] * test_all['Ins_Age']



#2 Multiplication of Weight and Age - higher the factor, higher the risk

test_all['age_wt'] = test_all['Ins_Age'] * test_all['Wt']



#3 Multiplication of Height and Age - higher the factor, higher the risk

test_all['age_ht'] = test_all['Ins_Age'] * test_all['Ht']



#4 Add all values of Medical Keywords Columns

med_keyword_columns = test_all.columns[test_all.columns.str.startswith('Medical_Keyword_')]

test_all['med_keywords_count'] = test_all[med_keyword_columns].sum(axis=1)

test_all.drop(med_keyword_columns, axis=1, inplace=True)



#5 BMI Categorization

conditions = [

    (test_all['BMI'] <= test_all['BMI'].quantile(0.25)),

    (test_all['BMI'] > test_all['BMI'].quantile(0.25)) & (test_all['BMI'] <= test_all['BMI'].quantile(0.75)),

    (test_all['BMI'] > test_all['BMI'].quantile(0.75))]

choices = ['under_weight', 'average', 'overweight']

test_all['bmi_wt'] = np.select(conditions, choices)



#6 Age Categorization

conditions = [

    (test_all['Ins_Age'] <= test_all['Ins_Age'].quantile(0.25)),

    (test_all['Ins_Age'] > test_all['Ins_Age'].quantile(0.25)) & (test_all['Ins_Age'] <= test_all['Ins_Age'].quantile(0.75)),

    (test_all['Ins_Age'] > test_all['Ins_Age'].quantile(0.75))]

choices = ['young', 'average', 'old']

test_all['age_cat'] = np.select(conditions, choices)



#7 Height Categorization

conditions = [

    (test_all['Ht'] <= test_all['Ht'].quantile(0.25)),

    (test_all['Ht'] > test_all['Ht'].quantile(0.25)) & (test_all['Ht'] <= test_all['Ht'].quantile(0.75)),

    (test_all['Ht'] > test_all['Ht'].quantile(0.75))]

choices = ['short', 'average', 'tall']

test_all['ht_cat'] = np.select(conditions, choices)



#8 Weight Categorization

conditions = [

    (test_all['Wt'] <= test_all['Wt'].quantile(0.25)),

    (test_all['Wt'] > test_all['Wt'].quantile(0.25)) & (test_all['Wt'] <= test_all['Wt'].quantile(0.75)),

    (test_all['Wt'] > test_all['Wt'].quantile(0.75))]

choices = ['thin', 'average', 'fat']

test_all['wt_cat'] = np.select(conditions, choices)
def new_target(row):

    if (row['bmi_wt']=='overweight') or (row['age_cat']=='old')  or (row['wt_cat']=='fat'):

        val='extremely_risky'

    else:

        val='not_extremely_risky'

    return val



df_all['extreme_risk'] = df_all.apply(new_target,axis=1)

test_all['extreme_risk'] = test_all.apply(new_target,axis=1)
df_all.extreme_risk.value_counts()
# Risk Categorization

conditions1 = [

    (df_all['bmi_wt'] == 'overweight') ,

    (df_all['bmi_wt'] == 'average') ,

    (df_all['bmi_wt'] == 'under_weight') ]

conditions2 = [

    (test_all['bmi_wt'] == 'overweight') ,

    (test_all['bmi_wt'] == 'average') ,

    (test_all['bmi_wt'] == 'under_weight') ]

choices = ['risk', 'non-risk', 'risk']

df_all['bmi_risk'] = np.select(conditions1, choices)

test_all['bmi_risk'] = np.select(conditions2, choices)
df_all.bmi_risk.value_counts()
def new_target(row):

    if (row['bmi_wt']=='average') or (row['age_cat']=='average')  or (row['wt_cat']=='average') or (row['ht_cat']=='average'):

        val='average'

    else:

        val='non_average'

    return val



df_all['average_risk'] = df_all.apply(new_target,axis=1)

test_all['average_risk'] = test_all.apply(new_target,axis=1)
df_all.average_risk.value_counts()
def new_target(row):

    if (row['bmi_wt']=='under_weight') or (row['age_cat']=='young')  or (row['wt_cat']=='thin') or (row['ht_cat']=='short'):

        val='low_end'

    else:

        val='non_low_end'

    return val

df_all['low_end_risk'] = df_all.apply(new_target,axis=1)

test_all['low_end_risk'] = test_all.apply(new_target,axis=1)
df_all.low_end_risk.value_counts()
def new_target(row):

    if (row['bmi_wt']=='overweight') or (row['age_cat']=='old')  or (row['wt_cat']=='fat') or (row['ht_cat']=='tall'):

        val='high_end'

    else:

        val='non_high_end'

    return val

df_all['high_end_risk'] = df_all.apply(new_target,axis=1)

test_all['high_end_risk'] = test_all.apply(new_target,axis=1)
df_all.high_end_risk.value_counts()
df_all.shape
test_all.shape
print(df_all.isnull().values.any())

print(test_all.isnull().values.any())
df_all.columns
#Delting all values of Medical History Columns having Value is biased more than 81%

med_keyword_columns = test_all.columns[test_all.columns.str.startswith('Medical_History_')]

for col in med_keyword_columns:

    print("Dropping column: " +col)

    print(df_all[col].value_counts(normalize=True) * 100)

    

drop_columns = ['Medical_History_3','Medical_History_5','Medical_History_6','Medical_History_7','Medical_History_9',

               'Medical_History_11','Medical_History_12','Medical_History_13','Medical_History_14',

               'Medical_History_16','Medical_History_17','Medical_History_18','Medical_History_19',

               'Medical_History_20','Medical_History_22','Medical_History_27','Medical_History_30',

               'Medical_History_31','Medical_History_33','Medical_History_34','Medical_History_35',

               'Medical_History_36','Medical_History_37','Medical_History_38','Medical_History_39',]     

df_all.drop(drop_columns, axis=1, inplace=True)

test_all.drop(drop_columns, axis=1, inplace=True)

print(df_all.shape)

print(test_all.shape)
plt.figure(figsize=(12,10))

sns.countplot(x = 'extreme_risk', hue = 'Response', data = df_all)
plt.figure(figsize=(12,10))

sns.countplot(x = 'average_risk', hue = 'Response', data = df_all)
plt.figure(figsize=(12,10))

sns.countplot(x = 'low_end_risk', hue = 'Response', data = df_all)
plt.figure(figsize=(12,10))

sns.countplot(x = 'high_end_risk', hue = 'Response', data = df_all)
plt.figure(figsize=(12,10))

sns.countplot(x = 'bmi_wt', hue = 'Response', data = df_all)
plt.figure(figsize=(12,10))

sns.countplot(x = 'age_cat', hue = 'Response', data = df_all)
plt.figure(figsize=(12,10))

sns.countplot(x = 'wt_cat', hue = 'Response', data = df_all)
plt.figure(figsize=(12,10))

sns.countplot(x = 'ht_cat', hue = 'Response', data = df_all)
plt.figure(figsize=(12,10))

sns.countplot(x = 'bmi_risk', hue = 'Response', data = df_all)
df_all.shape
test_all.shape
print(df_all.isnull().values.any())

print(test_all.isnull().values.any())
df_train = df_all

df_test = test_all
df_train.head()
df_train.drop(['Id'], axis=1, inplace=True)
df_train.dtypes
df_train.columns
# Categorical boolean mask

categorical_feature_mask = df_train.dtypes=='object'

# filter categorical columns using mask and turn it into a list

categorical_cols = df_train.columns[categorical_feature_mask].tolist()

categorical_cols
# import labelencoder

from sklearn.preprocessing import LabelEncoder

# instantiate labelencoder object

le = LabelEncoder()
# apply le on categorical feature columns

df_train[categorical_cols] = df_train[categorical_cols].apply(lambda col: le.fit_transform(col))

df_train[categorical_cols].head(10)
df_test[categorical_cols] = df_test[categorical_cols].apply(lambda col: le.fit_transform(col))

df_test[categorical_cols].head(10)
correlation(df_train,df_test,0.90)
X = df_train.drop(['Response'], axis=1)

y = df_train['Response']



# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
gc.collect()
# Create the parameter grid based on the results of random search 

param_grid = {

    'max_depth': [8,12,16]

    #'min_samples_leaf': range(30, 50),

    #'min_samples_split': range(20, 40),

    #'n_estimators': [300,500], 

    #'max_features': [24,48]

}

# Create a based model

rf = RandomForestClassifier()

# Instantiate the grid search model

rf = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", return_train_score = True)



rf.fit(X_train, y_train)

print('We can get accuracy of ',rf.best_score_,' using ',rf.best_params_)

scores = rf.cv_results_



plt.figure()

plt.plot(scores["param_max_depth"], 

       scores["mean_train_score"], 

        label="training accuracy")

plt.plot(scores["param_max_depth"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("max_depth")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
# Create the parameter grid based on the results of random search 

param_grid = {

    #'max_depth': [8,12,16],

    'min_samples_leaf': range(20,100,10)

    #'min_samples_split': range(20, 40),

    #'n_estimators': [300,500], 

    #'max_features': [24,48]

}

# Create a based model

rf = RandomForestClassifier()

# Instantiate the grid search model

rf = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", return_train_score = True)



rf.fit(X_train, y_train)

print('We can get accuracy of ',rf.best_score_,' using ',rf.best_params_)

scores = rf.cv_results_



plt.figure()

plt.plot(scores["param_min_samples_leaf"], 

       scores["mean_train_score"], 

        label="training accuracy")

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("min_samples_leaf")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
# Create the parameter grid based on the results of random search 

param_grid = {

    #'max_depth': [8,12,16],

    #'min_samples_leaf': range(20,100,10),

    'min_samples_split': range(20,200,10)

    #'n_estimators': [300,500], 

    #'max_features': [24,48]

}

# Create a based model

rf = RandomForestClassifier()

# Instantiate the grid search model

rf = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", return_train_score = True)



rf.fit(X_train, y_train)

print('We can get accuracy of ',rf.best_score_,' using ',rf.best_params_)

scores = rf.cv_results_



plt.figure()

plt.plot(scores["param_min_samples_split"], 

       scores["mean_train_score"], 

        label="training accuracy")

plt.plot(scores["param_min_samples_split"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("min_samples_split")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
# Create the parameter grid based on the results of random search 

param_grid = {

    #'max_depth': [8,12,16],

    #'min_samples_leaf': range(20,100,10),

    #'min_samples_split': range(20,100,10)

    #'n_estimators': [200,300,400,500]

    'max_features': [10,20,30,40,50]

}

# Create a based model

rf = RandomForestClassifier()

# Instantiate the grid search model

rf = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", return_train_score = True)



rf.fit(X_train, y_train)

print('We can get accuracy of ',rf.best_score_,' using ',rf.best_params_)

scores = rf.cv_results_



plt.figure()

plt.plot(scores["param_max_features"], 

       scores["mean_train_score"], 

        label="training accuracy")

plt.plot(scores["param_max_features"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("max_features")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
rfc = RandomForestClassifier(bootstrap=True,

                             max_depth=12,

                             min_samples_leaf=30, 

                             min_samples_split=80,

                             max_features=30,

                             n_estimators=600)

rfc.fit(X_train,y_train)

predictions = rfc.predict(X_test)

print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))
gc.collect()
y_train.unique()
# As labeles need to be from (0,7] instead of (1 to 8) else xgb.train will fail

y_train_new = y_train - 1

y_test_new = y_test - 1
y_train_new.unique()
n_classes = len(np.unique(y_train))

n_classes
params1 = {

    'learning_rate': 0.02,

    'max_depth': 16, 

    'n_estimators':500,

    'subsample':0.9,

    'objective':'multi:softmax',

    'num_class': n_classes

}
params2 = {

    'silent': False, 

    'learning_rate': 0.6,  

    'colsample_bytree': 0.4,

    'subsample': 0.8,

    'objective': 'multi:softmax', 

    'num_class': n_classes,

    'n_estimators': 1000, 

    'reg_alpha': 0.3,

    'max_depth': 32, 

    'gamma': 10

}
dtrain = xgb.DMatrix(data=X_train, label=y_train_new)

dtest = xgb.DMatrix(data=X_test)





# fit model on training data

xgb_model = xgb.train(params1, dtrain)



pred = xgb_model.predict(dtest)

print(pred)



print(classification_report(y_test_new, pred))

print(confusion_matrix(y_test_new,pred))
dtrain = xgb.DMatrix(data=X_train, label=y_train_new)

dtest = xgb.DMatrix(data=X_test)





# fit model on training data

xgb_model = xgb.train(params2, dtrain)



pred = xgb_model.predict(dtest)

print(pred)



print(classification_report(y_test_new, pred))

print(confusion_matrix(y_test_new,pred))