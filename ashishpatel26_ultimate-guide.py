import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as scs

%matplotlib inline
plt.style.use('ggplot') # overall 'ggplot' style

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
col_description = pd.read_csv('HomeCredit_columns_description.csv',
                              index_col=0,encoding='cp1252')

def col_name(name, col_description=col_description):
    values = col_description[col_description['Row'] == name]['Description'].values
    print(name + ': \n')
    for i in values:
        print (i+'\n')
    table = col_description[col_description['Row'] == name]['Table'].values
    print('Table: {}\n'.format(table))
application_train = pd.read_csv('../input/application_train.csv') 
application_test = pd.read_csv('../input/application_test.csv')
application_train.corr().abs()>0.5
target = application_train["TARGET"].value_counts()
df = pd.DataFrame({'target': target.index,
                   'count': target.values
                  })
plt.figure(figsize = (6,6))
plt.title('1 - Default vs  0 - Non default')
sns.barplot(x = 'target', y="count", data=df);
plt.figure(figsize=(20,8))
sns.distplot(application_train['AMT_INCOME_TOTAL'], hist=False, color='blue', label = 'Individual', kde_kws={'clip': (0, 100000)})
sns.distplot(application_test['AMT_INCOME_TOTAL'], hist=False, color='red', label = 'Individual', kde_kws={'clip': (0, 100000)})
plt.figure(figsize=(20,8))
sns.distplot(np.log(application_train['AMT_INCOME_TOTAL'][application_train["TARGET"]==0]), hist=False, color='blue', label = 'Repaid')#, kde_kws={'clip': (0, 100000)})
sns.distplot(np.log(application_train['AMT_INCOME_TOTAL'][application_train["TARGET"]==1]), hist=False, color='red', label = 'Defaulted')#, kde_kws={'clip': (0, 100000)})
application_train.describe().T
# Missings
total = application_train.isnull().sum().sort_values(ascending = False)
percent = (application_train.isnull().sum()/application_train.isnull().count()*100).sort_values(ascending = False)
missing_application_train_data  = pd.concat([total, percent], axis=1,
                                            keys=['Total', 'Percent'])
missing_application_train_data.head(20)
sets = [application_train, application_test]
for i in sets:
    i["AGE"] = application_train["DAYS_BIRTH"]/-365
    i["AGESQ"] = application_train["AGE"]**2
for i in sets:
    i_dummies=pd.get_dummies(i[['CODE_GENDER', 'NAME_EDUCATION_TYPE',
                               'NAME_FAMILY_STATUS', 'NAME_TYPE_SUITE', 
                                'NAME_INCOME_TYPE']],drop_first=True)
    i[i_dummies.columns]=i_dummies
    #print(i_dummies.columns)

for i in sets:
    i_dummies=pd.get_dummies(i[['OCCUPATION_TYPE']],drop_first=True)
    i[i_dummies.columns]=i_dummies
    #print(i_dummies.columns)
for i in sets:
    i['employed'] = i['DAYS_EMPLOYED']!=365243 *1
    i['YEARS_EMPLOYED'] = i['DAYS_EMPLOYED']/-365
    i["INCOME_LOG"] = np.log(i["AMT_INCOME_TOTAL"])
    i['employed_years']=i['employed']* i['YEARS_EMPLOYED']
# application_train[['OCCUPATION_TYPE', 'NAME_EDUCATION_TYPE', 'SK_ID_CURR']].groupby(['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE']).count()


for i in sets:
    i_dummies=pd.get_dummies(i[['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
                            'NAME_HOUSING_TYPE']],drop_first=True)
    i[i_dummies.columns]=i_dummies
    i_dummies.columns
for i in sets:
    i['FLAG_OWN_CAR_CAR_AGE']=i['FLAG_OWN_CAR_Y']*i['OWN_CAR_AGE']
for i in sets:
    i_dummies=pd.get_dummies(i[['NAME_CONTRACT_TYPE']],drop_first=True)
    i[i_dummies.columns]=i_dummies
    i_dummies.columns
for i in sets:
    i["AMT_CREDIT_LOG"] = np.log(i["AMT_CREDIT"]) # this is the amount of the loan
    i["AMT_GOODS_PRICE_LOG"] =np.log(i['AMT_GOODS_PRICE']) # FOR CONS LOANS-- half million usd???
    i["AMT_ANNUITY_LOG"] =np.log(i['AMT_ANNUITY'])
    i['CREDIT_INCOME'] = i["AMT_CREDIT"]/i["AMT_INCOME_TOTAL"]
    i['ANNUITY_INCOME'] = i['AMT_ANNUITY']/i["AMT_INCOME_TOTAL"]
docs =  ['FLAG_DOCUMENT_2',  'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4',
 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']

for i in sets:
    i['DOCS'] = 0
    for d in docs:
        application_train['DOCS'] += application_train[d]
    
# application_train['DOCS']
application_train.head().T
application_test.head().T
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, auc, roc_curve
from sklearn.preprocessing import PolynomialFeatures, Imputer, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report
# for i in sorted(application_train.columns):
#     print(i)
df_train_1 = application_train[['TARGET','AGE', 'AGESQ','REGION_RATING_CLIENT_W_CITY',
                                'DAYS_EMPLOYED','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3',
                                'CODE_GENDER_M', 'NAME_EDUCATION_TYPE_Higher education',
                                'NAME_EDUCATION_TYPE_Incomplete higher',
                                'NAME_EDUCATION_TYPE_Lower secondary',
                                'NAME_EDUCATION_TYPE_Secondary / secondary special',
                                'NAME_FAMILY_STATUS_Married', 'NAME_FAMILY_STATUS_Separated',
                                'NAME_FAMILY_STATUS_Single / not married', 
                                'NAME_FAMILY_STATUS_Widow', 'NAME_TYPE_SUITE_Family',
                                'NAME_TYPE_SUITE_Group of people', 'NAME_TYPE_SUITE_Other_A',
                                'NAME_TYPE_SUITE_Other_B', 'NAME_TYPE_SUITE_Spouse, partner',
                                'NAME_TYPE_SUITE_Unaccompanied',
                                'NAME_INCOME_TYPE_Commercial associate',
                                'NAME_INCOME_TYPE_Pensioner',
                                'NAME_INCOME_TYPE_State servant', 'NAME_INCOME_TYPE_Student',
                                'NAME_INCOME_TYPE_Unemployed', 'NAME_INCOME_TYPE_Working',
                                'OCCUPATION_TYPE_Cleaning staff', 'OCCUPATION_TYPE_Cooking staff',
                                'OCCUPATION_TYPE_Core staff', 'OCCUPATION_TYPE_Drivers',
                                'OCCUPATION_TYPE_HR staff', 'OCCUPATION_TYPE_High skill tech staff',
                                'OCCUPATION_TYPE_IT staff', 'OCCUPATION_TYPE_Laborers',
                                'OCCUPATION_TYPE_Low-skill Laborers', 'OCCUPATION_TYPE_Managers',
                                'OCCUPATION_TYPE_Medicine staff',
                                'OCCUPATION_TYPE_Private service staff',
                                'OCCUPATION_TYPE_Realty agents', 'OCCUPATION_TYPE_Sales staff',
                                'OCCUPATION_TYPE_Secretaries', 'OCCUPATION_TYPE_Security staff',
                                'OCCUPATION_TYPE_Waiters/barmen staff',
                                'employed', 'employed_years',
                                'INCOME_LOG',
                                'FLAG_OWN_CAR_Y', 'FLAG_OWN_REALTY_Y',
                                'NAME_HOUSING_TYPE_House / apartment',
                                'NAME_HOUSING_TYPE_Municipal apartment',
                                'NAME_HOUSING_TYPE_Office apartment',
                                'NAME_HOUSING_TYPE_Rented apartment', 'NAME_HOUSING_TYPE_With parents',
                                'FLAG_OWN_CAR_CAR_AGE', 'NAME_CONTRACT_TYPE_Revolving loans',
                                'AMT_CREDIT_LOG', 'AMT_GOODS_PRICE_LOG', 'AMT_ANNUITY_LOG',
                                'CREDIT_INCOME', 'ANNUITY_INCOME', 'DOCS']]
X_train = df_train_1.drop('TARGET',axis=1).values
y_train = df_train_1['TARGET'].values
X_test = application_test[['AGE', 'AGESQ','REGION_RATING_CLIENT_W_CITY',
                                'DAYS_EMPLOYED','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3',
                                'CODE_GENDER_M', 'NAME_EDUCATION_TYPE_Higher education',
                                'NAME_EDUCATION_TYPE_Incomplete higher',
                                'NAME_EDUCATION_TYPE_Lower secondary',
                                'NAME_EDUCATION_TYPE_Secondary / secondary special',
                                'NAME_FAMILY_STATUS_Married', 'NAME_FAMILY_STATUS_Separated',
                                'NAME_FAMILY_STATUS_Single / not married',
                                'NAME_FAMILY_STATUS_Widow', 'NAME_TYPE_SUITE_Family',
                                'NAME_TYPE_SUITE_Group of people', 'NAME_TYPE_SUITE_Other_A',
                                'NAME_TYPE_SUITE_Other_B', 'NAME_TYPE_SUITE_Spouse, partner',
                                'NAME_TYPE_SUITE_Unaccompanied',
                                'NAME_INCOME_TYPE_Commercial associate',
                                'NAME_INCOME_TYPE_Pensioner',
                                'NAME_INCOME_TYPE_State servant', 'NAME_INCOME_TYPE_Student',
                                'NAME_INCOME_TYPE_Unemployed', 'NAME_INCOME_TYPE_Working',
                                'OCCUPATION_TYPE_Cleaning staff', 'OCCUPATION_TYPE_Cooking staff',
                                'OCCUPATION_TYPE_Core staff', 'OCCUPATION_TYPE_Drivers',
                                'OCCUPATION_TYPE_HR staff', 'OCCUPATION_TYPE_High skill tech staff',
                                'OCCUPATION_TYPE_IT staff', 'OCCUPATION_TYPE_Laborers',
                                'OCCUPATION_TYPE_Low-skill Laborers', 'OCCUPATION_TYPE_Managers',
                                'OCCUPATION_TYPE_Medicine staff',
                                'OCCUPATION_TYPE_Private service staff',
                                'OCCUPATION_TYPE_Realty agents', 'OCCUPATION_TYPE_Sales staff',
                                'OCCUPATION_TYPE_Secretaries', 'OCCUPATION_TYPE_Security staff',
                                'OCCUPATION_TYPE_Waiters/barmen staff',
                                'employed', 'employed_years',
                                'INCOME_LOG',
                                'FLAG_OWN_CAR_Y', 'FLAG_OWN_REALTY_Y',
                                'NAME_HOUSING_TYPE_House / apartment',
                                'NAME_HOUSING_TYPE_Municipal apartment',
                                'NAME_HOUSING_TYPE_Office apartment',
                                'NAME_HOUSING_TYPE_Rented apartment', 'NAME_HOUSING_TYPE_With parents',
                                'FLAG_OWN_CAR_CAR_AGE', 'NAME_CONTRACT_TYPE_Revolving loans',
                                'AMT_CREDIT_LOG', 'AMT_GOODS_PRICE_LOG', 'AMT_ANNUITY_LOG',
                                'CREDIT_INCOME', 'ANNUITY_INCOME', 'DOCS']]
X_test = X_test.values
imp = Imputer(strategy='median') 
imp.fit(X_train) 

# transform the test & train data
X_train=imp.transform(X_train)
X_test=imp.transform(X_test)


scaler = MinMaxScaler(feature_range = (0, 1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print('Training data shape: ', X_train.shape)
print('Testing data shape: ', X_test.shape)
model = LogisticRegression(class_weight='balanced')
model.fit(X_train,y_train)
p = model.predict_proba(X_train)[:,1]
probabilities = model.predict_proba(X_test)[:,1]
# classification_report(y_train,p)
from statsmodels.discrete.discrete_model import Logit
model_sts = Logit(y_train,X_train)
model_sts.fit(y_train,X_train)
submit = application_test[['SK_ID_CURR']]
submit['TARGET'] = probabilities
submit.head()
submit.to_csv('logistic_2018-06-23.csv', index = False)
