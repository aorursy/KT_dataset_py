# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

#importing all the data files to dataset.
df1 = pd.read_csv('../input/application_test.csv')
df2 = pd.read_csv('../input/application_train.csv')
df3 = pd.read_csv('../input/bureau.csv')
df4 = pd.read_csv('../input/bureau_balance.csv')
df5 = pd.read_csv('../input/credit_card_balance.csv')
df6 = pd.read_csv('../input/installments_payments.csv')
df7 = pd.read_csv('../input/POS_CASH_balance.csv')
df8 = pd.read_csv('../input/previous_application.csv')

## check if column has more than 30% data missing, then reject that variables.
def accept_reject(data):
    total = data.isnull().sum().sort_values(ascending = False)
    NaN_Percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    final = pd.concat([total,NaN_Percent], axis=1, keys=['Total', 'NaN_Percent'])
    not_required =  final.drop(final[final.NaN_Percent < 30].index)
    print('Missing Value Greater Than 30%, so variable has been dropped')
    return not_required
## plot the graph of all the Catagorical variables
def cat_graph(data):
    print(' Catogirical fileds vs its count graph')
    catag= list(data.select_dtypes(exclude=['int64','float']).columns)
    for col in catag:
        #plt.title(col)
        sns.countplot(y= data[col])
        plt.show()
## If data is missing, then fill Catgorical with MODE and Continous with MEDIAN
def misfill(data):
    catag= pd.DataFrame([data.select_dtypes(exclude=['int64','float']).columns])
    contn= pd.DataFrame([data.select_dtypes(exclude=['object']).columns])
    for row in contn.iteritems():
        col =row[1] 
        data[col]=data[col].fillna(data[col].median())
    for roww in catag.iteritems():
        col =roww[1] 
        data[col]=data[col].fillna(data[col].mode().loc[0])
## Pre processing Application Train (df1)
df1.drop_duplicates(['SK_ID_CURR'], keep='first',inplace=True)
df1.sort_values(by=['SK_ID_CURR'],inplace=True)

accept_reject(df1)

## based on above function result, deleting below variable as it has more than 30% data missing.

df1.drop(['COMMONAREA_MEDI' , 'COMMONAREA_AVG', 'COMMONAREA_MODE','NONLIVINGAPARTMENTS_MODE',
          'NONLIVINGAPARTMENTS_MEDI','NONLIVINGAPARTMENTS_AVG','FONDKAPREMONT_MODE',\
          'LIVINGAPARTMENTS_MEDI','LIVINGAPARTMENTS_MODE','LIVINGAPARTMENTS_AVG','FLOORSMIN_MEDI',\
          'FLOORSMIN_MODE','FLOORSMIN_AVG','YEARS_BUILD_MEDI','YEARS_BUILD_AVG',\
          'YEARS_BUILD_MODE','OWN_CAR_AGE','LANDAREA_MODE','LANDAREA_AVG','LANDAREA_MEDI',\
          'BASEMENTAREA_MEDI','BASEMENTAREA_AVG','BASEMENTAREA_MODE','EXT_SOURCE_1',\
          'NONLIVINGAREA_MEDI','NONLIVINGAREA_AVG','NONLIVINGAREA_MODE','ELEVATORS_MODE',\
          'ELEVATORS_AVG','ELEVATORS_MEDI','WALLSMATERIAL_MODE','APARTMENTS_MODE','APARTMENTS_AVG',\
          'APARTMENTS_MEDI','ENTRANCES_MEDI','ENTRANCES_MODE','ENTRANCES_AVG','LIVINGAREA_MEDI',\
          'LIVINGAREA_MODE','LIVINGAREA_AVG','HOUSETYPE_MODE','FLOORSMAX_MODE','FLOORSMAX_MEDI',\
          'FLOORSMAX_AVG','YEARS_BEGINEXPLUATATION_MEDI','YEARS_BEGINEXPLUATATION_AVG',\
          'YEARS_BEGINEXPLUATATION_MODE','TOTALAREA_MODE','EMERGENCYSTATE_MODE',\
          'OCCUPATION_TYPE'],axis=1, inplace=True)

## plot the graphs for catagorical Variables

cat_graph(df1)

## fill the missing value 

misfill(df1)

# convert the string data to binary data.

df1 = pd.get_dummies(df1)

#preprocessing Training data (DF2)

df2.drop_duplicates(['SK_ID_CURR'], keep='first',inplace=True)
df1.sort_values(by=['SK_ID_CURR'],inplace=True)

accept_reject(df2)

df2.drop(['COMMONAREA_MEDI' , 'COMMONAREA_AVG', 'COMMONAREA_MODE','NONLIVINGAPARTMENTS_MODE',
          'NONLIVINGAPARTMENTS_MEDI','NONLIVINGAPARTMENTS_AVG','FONDKAPREMONT_MODE',\
          'LIVINGAPARTMENTS_MEDI','LIVINGAPARTMENTS_MODE','LIVINGAPARTMENTS_AVG','FLOORSMIN_MEDI',\
          'FLOORSMIN_MODE','FLOORSMIN_AVG','YEARS_BUILD_MEDI','YEARS_BUILD_AVG',\
          'YEARS_BUILD_MODE','OWN_CAR_AGE','LANDAREA_MODE','LANDAREA_AVG','LANDAREA_MEDI',\
          'BASEMENTAREA_MEDI','BASEMENTAREA_AVG','BASEMENTAREA_MODE','EXT_SOURCE_1',\
          'NONLIVINGAREA_MEDI','NONLIVINGAREA_AVG','NONLIVINGAREA_MODE','ELEVATORS_MODE',\
          'ELEVATORS_AVG','ELEVATORS_MEDI','WALLSMATERIAL_MODE','APARTMENTS_MODE','APARTMENTS_AVG',\
          'APARTMENTS_MEDI','ENTRANCES_MEDI','ENTRANCES_MODE','ENTRANCES_AVG','LIVINGAREA_MEDI',\
          'LIVINGAREA_MODE','LIVINGAREA_AVG','HOUSETYPE_MODE','FLOORSMAX_MODE','FLOORSMAX_MEDI',\
          'FLOORSMAX_AVG','YEARS_BEGINEXPLUATATION_MEDI','YEARS_BEGINEXPLUATATION_AVG',\
          'YEARS_BEGINEXPLUATATION_MODE','TOTALAREA_MODE','EMERGENCYSTATE_MODE',\
          'OCCUPATION_TYPE'],axis=1, inplace=True)

cat_graph(df2)

misfill(df2)

df2 = pd.get_dummies(df2)
# Pre processing the Bureau Dataset (df3)

df3.drop_duplicates(['SK_ID_BUREAU','SK_ID_CURR'], keep='first',inplace=True)
df3.sort_values(by=['SK_ID_BUREAU','SK_ID_CURR'],inplace=True)

accept_reject(df3)

#Based on above result, delete below variables
df3.drop(['CREDIT_CURRENCY', 'AMT_ANNUITY','AMT_CREDIT_MAX_OVERDUE','DAYS_ENDDATE_FACT',\
          'AMT_CREDIT_SUM_LIMIT','CREDIT_ACTIVE'], axis=1, inplace=True)

cat_graph(df3)

misfill(df3)

df3 = pd.get_dummies(df3) 


# Pre processing the Bureau Balance Dataset (df4)

df4.drop_duplicates(['SK_ID_BUREAU'], keep='first',inplace=True)
df4.sort_values(by=['SK_ID_BUREAU'],inplace=True)

accept_reject(df4)

cat_graph(df4)

misfill(df4)

df4 = pd.get_dummies(df4) 


# Pre processing Credit card balance Dataset (df5)

df5.drop_duplicates(['SK_ID_PREV','SK_ID_CURR'], keep='first',inplace=True)
df5.sort_values(by=['SK_ID_PREV','SK_ID_CURR'],inplace=True)

accept_reject(df5)

df5.drop(['NAME_CONTRACT_STATUS','AMT_RECEIVABLE_PRINCIPAL','AMT_RECIVABLE',\
          'AMT_TOTAL_RECEIVABLE','AMT_PAYMENT_CURRENT','AMT_DRAWINGS_ATM_CURRENT',\
         'CNT_DRAWINGS_POS_CURRENT','CNT_DRAWINGS_OTHER_CURRENT','CNT_DRAWINGS_ATM_CURRENT',\
         'AMT_DRAWINGS_POS_CURRENT','AMT_DRAWINGS_OTHER_CURRENT'],axis=1, inplace=True)

cat_graph(df5)

misfill(df5)

df5 = pd.get_dummies(df5)

## Pre processing Instalment_Payment datasets (df6)

df6.drop_duplicates(['SK_ID_PREV','SK_ID_CURR'], keep='first',inplace=True)
df6.sort_values(by=['SK_ID_PREV','SK_ID_CURR'],inplace=True)

accept_reject(df6)

cat_graph(df6)

misfill(df6)

df6 = pd.get_dummies(df6)
## Pre processing Posh Cash Balance dataset (df7)

df7.drop_duplicates(['SK_ID_PREV','SK_ID_CURR'], keep='first',inplace=True)
df7.sort_values(by=['SK_ID_PREV','SK_ID_CURR'],inplace=True)

accept_reject(df7)

cat_graph(df7)

misfill(df7)

df7 = pd.get_dummies(df7)

## Pre processing Previous Applicataion dataset (df8)

df8.drop_duplicates(['SK_ID_PREV','SK_ID_CURR'], keep='first',inplace=True)
df8.sort_values(by=['SK_ID_PREV','SK_ID_CURR'],inplace=True)

accept_reject(df8)

# Drop below fileds based on above function result.
df8.drop(['RATE_INTEREST_PRIVILEGED','RATE_INTEREST_PRIMARY','RATE_DOWN_PAYMENT','AMT_DOWN_PAYMENT',\
          'NAME_TYPE_SUITE', 'DAYS_TERMINATION','NFLAG_INSURED_ON_APPROVAL','DAYS_FIRST_DRAWING',\
          'DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE'], axis=1, inplace=True)

cat_graph(df8)

df8 = pd.get_dummies(df8)
# MERGE  beauro and beauro balance on 'SK_ID_BUREAU'

df34= pd.merge(df3, df4, on='SK_ID_BUREAU')

# Merge credit card balance and installation payment on 'SK_ID_PREV','SK_ID_CURR'

df56= pd.merge(df5, df6, on=['SK_ID_PREV','SK_ID_CURR'],how='inner')

#pos cash balance is not matching at all with above tables, so ignored that complete table.

# Merge prev application, credit card balance and installation payment on 'SK_ID_PREV','SK_ID_CURR'

df856 = pd.merge(df8, df56, on=['SK_ID_PREV','SK_ID_CURR'])

## Merge beauro,beauro balance,prev application, credit card balance and installation payment on 'SK_ID_CURR'

df_sub_final= pd.merge(df34, df856, on='SK_ID_CURR')

# Merge the Training dataset with all other dataset on 'SK_ID_CURR'
actual_train_data = pd.merge(df2, df_sub_final, on='SK_ID_CURR',how = 'inner')

# Merge the Test dataset with all other dataset on 'SK_ID_CURR'

actual_test_data = pd.merge(df1, df_sub_final, on='SK_ID_CURR',how = 'left')

# to match the variables in test and train data. Below variable has to be removed as these fields are not in Test dataset 

actual_train_data.drop(['CODE_GENDER_XNA','NAME_INCOME_TYPE_Maternity leave','NAME_FAMILY_STATUS_Unknown'],axis=1,inplace=True)

misfill(actual_test_data)
misfill(actual_train_data)
actual_test_data = pd.get_dummies(actual_test_data)
actual_train_data = pd.get_dummies(actual_train_data)
# PREDICT THE BEST FIT MODEL
# first preditict using the Training data (as target also given to check accuracy), then we will fit the best model in Test.

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split

y = actual_train_data['TARGET']
X=  actual_train_data.drop(['TARGET'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



#1. DECISION TREE CLASSIFIER 
from sklearn.tree import DecisionTreeClassifier 
Dtree = DecisionTreeClassifier()
dt= Dtree.fit(X_train,y_train)
predict = dt.predict(X_test)
dt_acc = accuracy_score(y_test, predict)
dt_conf = confusion_matrix(y_test, predict)
print('Decision Tree accuracy',dt_acc)
print('Decision Tree Confusion Matrix',dt_conf)

#2. RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_prediction = random_forest.predict(X_test)
random_acc = accuracy_score(y_test, Y_prediction)
random_conf = confusion_matrix(y_test, Y_prediction)
print('Random Forest accuracy',random_acc)
print('Random Forest Confusion Matrix',random_conf)

#3. K-Neighbour Classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
Y_knn_Pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, Y_knn_Pred)
knn_conf = confusion_matrix(y_test, Y_knn_Pred)
print('K-Neighbour accuracy',knn_acc)
print('K-Neighbour Confusion Matrix',knn_conf)

#4. LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_log_Pred = logreg.predict(X_test)
logistic_acc = accuracy_score(y_test, Y_log_Pred)
logistic_conf = confusion_matrix(y_test, Y_log_Pred)
print('Logistic accuracy',logistic_acc)
print('Logistic Confusion Matrix',logistic_conf)



## APPLY THE BEST MODEL ON "ACTUAL TEST DATA " - (DF1)

# Best accuracy we got for DecisionTreeClassifier, so we can apply the same to our actual test data

pred = dt.predict(actual_test_data)

pred = pd.DataFrame(pred)

Result = pd.concat([actual_test_data['SK_ID_CURR'], pred], axis=1)

Result