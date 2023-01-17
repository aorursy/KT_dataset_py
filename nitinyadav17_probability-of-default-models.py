import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_profiling as pp
import os,warnings
from scipy.stats import skew
import seaborn as sns
import pandas_profiling as pp
from statistics import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance
df_train = pd.read_csv('../input/home-credit-default-risk/application_train.csv')
df_train['TARGET'].value_counts(normalize = True)*100
df_train.head(10)
df0 = df_train[df_train.TARGET == 0]
df0.shape
def missing(df):
    value = df.isnull().sum()
    value_per = 100*df.isnull().sum()/len(df)
    missing_values = pd.concat([value,value_per], axis = 1)
    missing_values = missing_values.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'}).sort_values('% of Total Values',ascending = False)
    missing_values = missing_values[missing_values.iloc[:,1] !=0].round(1)
    return missing_values
x_missing =missing(df0)
x_missing = x_missing[x_missing['% of Total Values']> 45]
x_missing
df0 = df0.drop(df0[df0['OWN_CAR_AGE'].isnull()].index)
df1 = df_train[df_train['TARGET'] == 1]
df_train = pd.concat([df0,df1])
df_train.shape
Report = pp.ProfileReport(df_train,minimal = True)
df_train = df_train.drop(df_train[df_train['CODE_GENDER'] == 'XNA'].index)
df_train = df_train.drop(df_train[df_train['CNT_CHILDREN']>3].index)
df_train = df_train.drop(df_train[df_train['AMT_INCOME_TOTAL']>1145000].index)
df_train = df_train.drop(df_train[df_train['CNT_FAM_MEMBERS']>6].index)
df_train = df_train.drop(df_train[df_train['CNT_FAM_MEMBERS'].isnull()].index)
df_train = df_train.drop(df_train[df_train['NAME_TYPE_SUITE'].isnull()].index)
df_train = df_train.drop(df_train[df_train['NAME_FAMILY_STATUS'] == 'Unknown'].index)
df_train = df_train.drop(df_train[df_train['REGION_POPULATION_RELATIVE']>0.0623].index)
df_train = df_train.drop(df_train[df_train['DAYS_EMPLOYED'] == 365243].index)
df_train['OCCUPATION_TYPE'].fillna('U', inplace = True)
df_train["OWN_CAR_AGE"] = np.where(df_train['OWN_CAR_AGE']>40,40,df_train['OWN_CAR_AGE'])
df_train = df_train.drop(df_train[df_train['EXT_SOURCE_2'].isnull()].index)
df_train = df_train.drop(df_train[df_train['DAYS_LAST_PHONE_CHANGE'].isnull()].index)
df_train["DAYS_BIRTH"] = abs(df_train["DAYS_BIRTH"])
df_train['DAYS_EMPLOYED'] = abs(df_train['DAYS_EMPLOYED'])
df_train = df_train.drop(df_train[df_train['DEF_30_CNT_SOCIAL_CIRCLE'].isnull()].index)
df_train = df_train.drop(df_train[df_train['DEF_60_CNT_SOCIAL_CIRCLE'].isnull()].index)
df_train = df_train.drop(df_train[df_train['OBS_30_CNT_SOCIAL_CIRCLE'].isnull()].index)
df_train = df_train.drop(df_train[df_train['OBS_60_CNT_SOCIAL_CIRCLE'].isnull()].index)
df_train.shape
def missing(df):
    value = df.isnull().sum()
    value_per = 100*df.isnull().sum()/len(df)
    missing_values = pd.concat([value,value_per], axis = 1)
    missing_values = missing_values.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values',2:'Skewness'}).sort_values('% of Total Values',ascending = False)
    missing_values = missing_values[missing_values.iloc[:,1] !=0].round(1)
    return missing_values
x_missing =missing(df_train)
x_missing = x_missing[x_missing['% of Total Values']> 45]
x_missing
x_missing.index
df_train = df_train.drop(columns=['COMMONAREA_AVG', 'COMMONAREA_MODE', 'COMMONAREA_MEDI',
       'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAPARTMENTS_MODE',
       'NONLIVINGAPARTMENTS_MEDI', 'LIVINGAPARTMENTS_AVG',
       'LIVINGAPARTMENTS_MEDI', 'LIVINGAPARTMENTS_MODE', 'FONDKAPREMONT_MODE',
       'FLOORSMIN_AVG', 'FLOORSMIN_MODE', 'FLOORSMIN_MEDI', 'YEARS_BUILD_AVG',
       'YEARS_BUILD_MEDI', 'YEARS_BUILD_MODE', 'LANDAREA_MEDI', 'LANDAREA_AVG',
       'LANDAREA_MODE', 'BASEMENTAREA_MEDI', 'BASEMENTAREA_AVG',
       'BASEMENTAREA_MODE', 'NONLIVINGAREA_AVG', 'NONLIVINGAREA_MODE',
       'NONLIVINGAREA_MEDI', 'ELEVATORS_MODE', 'ELEVATORS_AVG',
       'ELEVATORS_MEDI', 'WALLSMATERIAL_MODE', 'APARTMENTS_AVG',
       'APARTMENTS_MODE', 'APARTMENTS_MEDI', 'ENTRANCES_MODE', 'ENTRANCES_AVG',
       'ENTRANCES_MEDI', 'HOUSETYPE_MODE', 'LIVINGAREA_MODE', 'LIVINGAREA_AVG',
       'LIVINGAREA_MEDI', 'FLOORSMAX_MODE', 'FLOORSMAX_MEDI', 'FLOORSMAX_AVG',
       'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BEGINEXPLUATATION_MEDI',
       'YEARS_BEGINEXPLUATATION_MODE', 'TOTALAREA_MODE',
       'EMERGENCYSTATE_MODE'],axis=1)
df_train.shape
df_int = df_train.select_dtypes('int64')
df_int.shape
df_flo = df_train.select_dtypes('float')
df_flo.columns
df_flo['SK_ID_CURR'] = df_train['SK_ID_CURR']
df_obj = df_train.select_dtypes('object')
df_obj.shape
df_flo.isnull().sum().sort_values(ascending = False)
df_flo.columns
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(df_flo)
df_flow = imputer.transform(df_flo)

df_flow = pd.DataFrame(df_flow,columns=['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
       'REGION_POPULATION_RELATIVE', 'DAYS_REGISTRATION', 'OWN_CAR_AGE',
       'CNT_FAM_MEMBERS', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
       'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
       'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
       'DAYS_LAST_PHONE_CHANGE', 'AMT_REQ_CREDIT_BUREAU_HOUR',
       'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
       'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
       'AMT_REQ_CREDIT_BUREAU_YEAR', 'SK_ID_CURR'])
df_flow.isnull().sum().sort_values(ascending = False)
df_intflow = pd.merge(df_flow,df_int,how = 'inner',on = 'SK_ID_CURR')
df_intflow.shape
df_obj['SK_ID_CURR'] = df_train['SK_ID_CURR']
df_obj.shape
df_final = pd.merge(df_obj,df_intflow)
df_final.isnull().sum()
df_final.columns
df_final = df_final.reindex(columns=['SK_ID_CURR','NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
       'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
       'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 
       'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
       'REGION_POPULATION_RELATIVE', 'DAYS_REGISTRATION', 'CNT_FAM_MEMBERS',
       'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
       'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
       'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
       'DAYS_LAST_PHONE_CHANGE', 'AMT_REQ_CREDIT_BUREAU_HOUR',
       'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
       'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
       'AMT_REQ_CREDIT_BUREAU_YEAR',  'CNT_CHILDREN', 'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH', 'FLAG_MOBIL', 'FLAG_EMP_PHONE',
       'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
       'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
       'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION',
       'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
       'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
       'LIVE_CITY_NOT_WORK_CITY', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',
       'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
       'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
       'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
       'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
       'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21','TARGET'])
df_final.shape
df_final['TARGET'].value_counts()
X = df_final.iloc[:,:-1]
y = df_final.iloc[:,-1]
X = pd.DataFrame(X,columns=['SK_ID_CURR','NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
       'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
       'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 
       'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
       'REGION_POPULATION_RELATIVE', 'DAYS_REGISTRATION', 'CNT_FAM_MEMBERS',
       'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
       'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
       'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
       'DAYS_LAST_PHONE_CHANGE', 'AMT_REQ_CREDIT_BUREAU_HOUR',
       'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
       'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
       'AMT_REQ_CREDIT_BUREAU_YEAR',  'CNT_CHILDREN', 'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH', 'FLAG_MOBIL', 'FLAG_EMP_PHONE',
       'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
       'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
       'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION',
       'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
       'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
       'LIVE_CITY_NOT_WORK_CITY', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',
       'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
       'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
       'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
       'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
       'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21'])

fig, ax = plt.subplots(figsize=(20,10))         
corr = X.corr()
sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)
ax.set_title("Correlation Matrix", fontsize=14)
plt.show()
X = X.drop(columns = ['CNT_CHILDREN','SK_ID_CURR'])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in X:
    if X[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(X[col].unique())) <= 2:
            # Train on the training data
            le.fit(X[col])
            # Transform both training and testing data
            X[col] = le.transform(X[col])
            
X=pd.get_dummies(X,drop_first=True)
list(X.columns)
X = X.drop(columns=['NAME_TYPE_SUITE_Other_A','NAME_INCOME_TYPE_Working','NAME_EDUCATION_TYPE_Secondary / secondary special','NAME_FAMILY_STATUS_Widow','NAME_HOUSING_TYPE_With parents','OCCUPATION_TYPE_Waiters/barmen staff','WEEKDAY_APPR_PROCESS_START_WEDNESDAY','ORGANIZATION_TYPE_University'])
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 123)
X_train.shape
y_train.value_counts()
y_test.value_counts()
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0,solver='liblinear')
classifier.fit(X_train,y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(classification_report(y_test,y_pred))
print('AUPRC = {}'.format(roc_auc_score(y_test, y_pred)))
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(classification_report(y_test,y_pred))
print('AUPRC = {}'.format(roc_auc_score(y_test, y_pred)))
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(classification_report(y_test,y_pred))
print('AUPRC = {}'.format(roc_auc_score(y_test, y_pred)))
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(classification_report(y_test,y_pred))
print('AUPRC = {}'.format(roc_auc_score(y_test, y_pred)))
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

from xgboost.sklearn import XGBClassifier
classifier = XGBClassifier(n_estimators=35,
 max_depth= 3,
 max_delta_step = 26,
 learning_rate = 0.15,
 gamma = 0.1,
min_child_weight = 3)
classifier.fit(X_train, y_train)
from sklearn.model_selection import RandomizedSearchCV 
import xgboost as xgb
param_grid = {'n_estimators': [ 30, 35, 25],
                    'learning_rate': [ 0.1, 0.15,0.2],
                    'gamma':  [0.20,0.10, 0.15],
                    'max_delta_step': [24, 26, 22],
                    'max_depth':[4, 3, 5],
             'min_child_weight': [1, 2, 3, 4]}       

ransearch = RandomizedSearchCV(classifier, n_iter = 10, param_distributions=param_grid, cv=3, n_jobs=-1, verbose=2)
ransearch = ransearch.fit(X_train,y_train)
best_accuracy = ransearch.best_score_
best_parameter = ransearch.best_params_
print('Accuracy: {:.2f}%'.format(best_accuracy*100))
print('parameter:',best_parameter)

classifier = XGBClassifier(n_estimators=35,
 max_depth= 4,
 max_delta_step = 24,
 learning_rate = 0.2,
 gamma = 0.15,
min_child_weight = 2)
classifier.fit(X_train, y_train)
# Classification Report: -from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(classification_report(y_test,y_pred))
print('AUPRC = {}'.format(roc_auc_score(y_test, y_pred)))
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

disp = plot_precision_recall_curve(classifier, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
fig = plt.figure(figsize = (14, 9))
ax = fig.add_subplot(111)

colours = plt.cm.Set1(np.linspace(0, 1, 9))

ax = plot_importance(classifier, height = 1, color = colours, grid = False, \
                     show_values = False, importance_type = 'cover', ax = ax);
for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        
ax.set_xlabel('importance score', size = 16);
ax.set_ylabel('features', size = 16);
ax.set_yticklabels(ax.get_yticklabels(), size = 12);
ax.set_title('Ordering of features by importance to the model learnt', size = 20);


