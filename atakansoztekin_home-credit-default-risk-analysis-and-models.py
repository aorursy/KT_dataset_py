import pandas as pd
import numpy as np
import warnings
import math
import os

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
warnings.filterwarnings('ignore')
installment_payments = pd.read_csv("../input/installments_payments.csv")
installment_payments.dropna(inplace=True)

installment_payments_new = installment_payments.groupby(['SK_ID_CURR','SK_ID_PREV','NUM_INSTALMENT_NUMBER'],as_index=False).mean()
installment_payments_new['AMT_INSTALMENT'] = np.round(installment_payments.groupby(['SK_ID_CURR','SK_ID_PREV','NUM_INSTALMENT_NUMBER'],as_index=False).sum()['AMT_INSTALMENT'],decimals=3)

l = installment_payments_new[installment_payments_new.AMT_PAYMENT>installment_payments_new.AMT_INSTALMENT].index
installment_payments_new.drop(index=l,inplace=True)

installment_payments_new['payment'] = installment_payments_new.AMT_INSTALMENT-installment_payments_new.AMT_PAYMENT
installment_payments_new['days'] = installment_payments_new.iloc[:,4]-installment_payments_new.iloc[:,5]
early_max = np.min(installment_payments_new.DAYS_INSTALMENT-installment_payments_new.DAYS_ENTRY_PAYMENT)
installment_payments_new.days = installment_payments_new.days - early_max

installment_payments_new['score'] = 100 - (((installment_payments_new.days/np.max(installment_payments_new.days)) + (installment_payments_new.payment/np.max(installment_payments_new.payment)))/2*100)

score_last_installment = installment_payments_new.groupby(['SK_ID_CURR'],as_index=False)['score'].mean()
bureau = pd.read_csv("../input/bureau.csv")

bureau_new = bureau.groupby(['SK_ID_CURR','SK_ID_BUREAU','CREDIT_ACTIVE'],as_index=False).mean()
bureau_new['score'] = np.nan

l1 = bureau_new[bureau_new.CREDIT_DAY_OVERDUE!=0].index
bureau_new.loc[l1,'score'] =100-(((bureau_new.loc[l1,'CREDIT_DAY_OVERDUE']/np.max(bureau_new.loc[l1,'CREDIT_DAY_OVERDUE']))+(bureau_new.loc[l1,'AMT_CREDIT_SUM_OVERDUE']/np.max(bureau_new.loc[l1,'AMT_CREDIT_SUM_OVERDUE'])))/2*100) 

l2 = bureau_new[(bureau_new.CREDIT_DAY_OVERDUE==0)&(bureau_new.CREDIT_ACTIVE=='Closed')].index
divisionl2 = bureau_new.loc[l2,'AMT_CREDIT_SUM']/(bureau_new.loc[l2,'DAYS_CREDIT_ENDDATE']-bureau_new.loc[l2,'DAYS_CREDIT'])
divisionl2.replace([np.inf, -np.inf], np.nan,inplace=True)
bureau_new.loc[l2,'score'] = (divisionl2 / np.max(divisionl2))*100


l3 = bureau_new[(bureau_new.CREDIT_DAY_OVERDUE==0)&(bureau_new.CREDIT_ACTIVE=='Active')].index
divisionl3 = (bureau_new.loc[l3,'AMT_CREDIT_SUM']-bureau_new.loc[l3,'AMT_CREDIT_SUM_DEBT'])/(bureau_new.loc[l3,'DAYS_CREDIT_ENDDATE']-bureau_new.loc[l3,'DAYS_CREDIT'])
divisionl3.replace([np.inf, -np.inf], np.nan,inplace=True)
bureau_new.loc[l3,'score'] = (divisionl3 / np.max(divisionl3))*100

score_last_bureau = bureau_new.groupby(['SK_ID_CURR'],as_index=False)['score'].mean()
score_last_bureau.dropna(subset=['score'],inplace=True)
import matplotlib.pyplot as plt
import seaborn as sns

application_train = pd.read_csv("../input/application_train.csv")
application_test = pd.read_csv("../input/application_test.csv")
target_variable = application_train.TARGET
application_train.drop('TARGET',axis=1,inplace=True)

#Bureau and installment score matching
application_train['bureau_score']=np.nan
application_train['installment_score']=np.nan

l = list(score_last_bureau.SK_ID_CURR)
l_last = list(application_train[application_train.SK_ID_CURR.isin(l)]['SK_ID_CURR'])
application_train.loc[application_train.SK_ID_CURR.isin(l_last),'bureau_score']=score_last_bureau.loc[score_last_bureau.SK_ID_CURR.isin(l_last),'score'].values

l2 = list(score_last_installment.SK_ID_CURR)
l2_last = list(application_train[application_train.SK_ID_CURR.isin(l2)]['SK_ID_CURR'])
application_train.loc[application_train.SK_ID_CURR.isin(l2_last),'installment_score']=score_last_installment.loc[score_last_installment.SK_ID_CURR.isin(l2_last),'score'].values

application_test['bureau_score']=np.nan
application_test['installment_score']=np.nan

l = list(score_last_bureau.SK_ID_CURR)
l_last = list(application_test[application_test.SK_ID_CURR.isin(l)]['SK_ID_CURR'])
application_test.loc[application_test.SK_ID_CURR.isin(l_last),'bureau_score']=score_last_bureau.loc[score_last_bureau.SK_ID_CURR.isin(l_last),'score'].values

l2 = list(score_last_installment.SK_ID_CURR)
l2_last = list(application_test[application_test.SK_ID_CURR.isin(l2)]['SK_ID_CURR'])
application_test.loc[application_test.SK_ID_CURR.isin(l2_last),'installment_score']=score_last_installment.loc[score_last_installment.SK_ID_CURR.isin(l2_last),'score'].values
# %20 or more missing values handling
missing_values = pd.DataFrame(np.ones(shape=(123,2)),columns=['Columns','Total NaNs'])
missing_values['Columns'] = application_train.isnull().sum().index
missing_values['Total NaNs'] = application_train.isnull().sum().values

missing_values['NanP'] = missing_values['Total NaNs'] / len(application_train)

droped_columns = missing_values.loc[missing_values.NanP>0.2,'Columns']
application_train.drop(droped_columns,inplace=True,axis=1)
missing_values = pd.DataFrame(np.ones(shape=(123,2)),columns=['Columns','Total NaNs'])
missing_values['Columns'] = application_test.isnull().sum().index
missing_values['Total NaNs'] = application_test.isnull().sum().values

missing_values['NanP'] = missing_values['Total NaNs'] / len(application_test)

droped_columns = missing_values.loc[missing_values.NanP>0.2,'Columns']
application_test.drop(droped_columns,inplace=True,axis=1)
application_train.dtypes.value_counts()
application_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
# Categorical Variable Handling

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in application_train:
    if application_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(application_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(application_train[col])
            # Transform both training and testing data
            application_train[col] = le.transform(application_train[col])
            application_test[col] = le.transform(application_test[col])

# one-hot encoding of categorical variables
application_train = pd.get_dummies(application_train)
application_test = pd.get_dummies(application_test)

# %10 or less missing values handling
missing_values = pd.DataFrame(np.ones(shape=(165,2)),columns=['Columns','Total NaNs'])
missing_values['Columns'] = application_train.isnull().sum().index
missing_values['Total NaNs'] = application_train.isnull().sum().values

missing_values['NanP'] = missing_values['Total NaNs'] / len(application_train)

median_columns = missing_values[(missing_values.NanP>0)&(missing_values.NanP<=0.2)]['Columns']
for col in median_columns:
    application_train[col].fillna(application_train[col].median(),inplace=True)
# Categorical Variable Handling

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in application_test:
    if application_test[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(application_test[col].unique())) <= 2:
            # Train on the training data
            le.fit(application_test[col])
            # Transform both training and testing data
            application_test[col] = le.transform(application_test[col])
            application_test[col] = le.transform(application_test[col])

# one-hot encoding of categorical variables
application_test = pd.get_dummies(application_test)
application_test = pd.get_dummies(application_test)
# %10 or less missing values handling
missing_values = pd.DataFrame(np.ones(shape=(162,2)),columns=['Columns','Total NaNs'])
missing_values['Columns'] = application_test.isnull().sum().index
missing_values['Total NaNs'] = application_test.isnull().sum().values

missing_values['NanP'] = missing_values['Total NaNs'] / len(application_test)

median_columns = missing_values[(missing_values.NanP>0)&(missing_values.NanP<=0.2)]['Columns']
for col in median_columns:
    application_test[col].fillna(application_test[col].median(),inplace=True)
anomaly_df = application_train.describe()

anomaly_columns=anomaly_df.columns[anomaly_df.iloc[7,:]!=1]
anomaly_df[anomaly_columns]
25229/365
7489/365
application_train.DAYS_EMPLOYED = application_train.DAYS_EMPLOYED - np.min(application_train.DAYS_EMPLOYED)
application_test.DAYS_EMPLOYED = application_test.DAYS_EMPLOYED - np.min(application_test.DAYS_EMPLOYED)

anomaly_df = application_train.describe()

anomaly_columns=anomaly_df.columns[anomaly_df.iloc[7,:]!=1]
anomaly_df[anomaly_columns]
50*365
l = application_train[application_train.DAYS_EMPLOYED>18250]['DAYS_EMPLOYED'].index
application_train.loc[l,'DAYS_EMPLOYED'] = np.nan
application_train['DAYS_EMPLOYED'].fillna(application_train['DAYS_EMPLOYED'].median(),inplace=True)
l = application_test[application_test.DAYS_EMPLOYED>18250]['DAYS_EMPLOYED'].index
application_test.loc[l,'DAYS_EMPLOYED'] = np.nan
application_test['DAYS_EMPLOYED'].fillna(application_test['DAYS_EMPLOYED'].median(),inplace=True)
l = application_train[application_train.DAYS_REGISTRATION>18250]['DAYS_REGISTRATION'].index
application_train.loc[l,'DAYS_REGISTRATION'] = np.nan
application_train['DAYS_REGISTRATION'].fillna(application_train['DAYS_REGISTRATION'].median(),inplace=True)
l = application_test[application_test.DAYS_REGISTRATION>18250]['DAYS_REGISTRATION'].index
application_test.loc[l,'DAYS_REGISTRATION'] = np.nan
application_test['DAYS_REGISTRATION'].fillna(application_test['DAYS_REGISTRATION'].median(),inplace=True)
application_train['TARGET'] = target_variable
corr = application_train.corr()['TARGET'].sort_values()

corr.tail(3)
corr.head(2)
plt.figure(figsize = (6, 4))

# KDE plot of loans that were repaid on time
sns.kdeplot(application_train.loc[application_train['TARGET'] == 0, 'EXT_SOURCE_2'], label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(application_train.loc[application_train['TARGET'] == 1, 'EXT_SOURCE_2'], label = 'target == 1')

# Labeling of plot
plt.xlabel('EXT_SOURCE_2'); plt.ylabel('Density'); plt.title('Distribution EXT_SOURCE');
plt.figure(figsize = (6, 4))

# KDE plot of loans that were repaid on time
sns.kdeplot(application_train.loc[application_train['TARGET'] == 0, 'DAYS_EMPLOYED'] / 365, label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(application_train.loc[application_train['TARGET'] == 1, 'DAYS_EMPLOYED'] / 365, label = 'target == 1')

# Labeling of plot
plt.xlabel('DAYS_EMPLOYED'); plt.ylabel('Density'); plt.title('Distribution DAYS_EMPLOYED');
application_train_new = application_train.copy()
application_test_new = application_test.copy()

application_train_new['INCOME_PER_PERSON'] = application_train['AMT_INCOME_TOTAL'] / application_train['CNT_FAM_MEMBERS']
application_train_new['CREDIT_INCOME_PERCENT'] = application_train_new['AMT_CREDIT'] / application_train_new['AMT_INCOME_TOTAL']
application_train_new['ANNUITY_INCOME_PERCENT'] = application_train_new['AMT_ANNUITY'] / application_train_new['AMT_INCOME_TOTAL']
application_train_new['CREDIT_TERM'] = application_train_new['AMT_ANNUITY'] / application_train_new['AMT_CREDIT']
application_train_new['DAYS_EMPLOYED_PERCENT'] = application_train_new['DAYS_EMPLOYED'] / application_train_new['DAYS_BIRTH']
application_train_new['GOOD_PRICE_PERCENT'] = application_train_new['AMT_GOODS_PRICE'] / application_train_new['AMT_INCOME_TOTAL']
application_train_new['CHILDREN_PERCENT'] = application_train_new['CNT_CHILDREN'] / application_train_new['CNT_FAM_MEMBERS']
application_train_new['OBS_30_CNT_SOCIAL_CIRCLE_REGION'] = application_train_new.OBS_30_CNT_SOCIAL_CIRCLE * application_train_new.REGION_POPULATION_RELATIVE
application_train_new['DEF_30_CNT_SOCIAL_CIRCLE_REGION'] = application_train_new.DEF_30_CNT_SOCIAL_CIRCLE * application_train_new.REGION_POPULATION_RELATIVE
application_train_new['OBS_60_CNT_SOCIAL_CIRCLE_REGION'] = application_train_new.OBS_60_CNT_SOCIAL_CIRCLE * application_train_new.REGION_POPULATION_RELATIVE
application_train_new['DEF_60_CNT_SOCIAL_CIRCLE_REGION'] = application_train_new.DEF_60_CNT_SOCIAL_CIRCLE * application_train_new.REGION_POPULATION_RELATIVE

application_test_new['INCOME_PER_PERSON'] = application_train['AMT_INCOME_TOTAL'] / application_train['CNT_FAM_MEMBERS']
application_test_new['CREDIT_INCOME_PERCENT'] = application_test_new['AMT_CREDIT'] / application_test_new['AMT_INCOME_TOTAL']
application_test_new['ANNUITY_INCOME_PERCENT'] = application_test_new['AMT_ANNUITY'] / application_test_new['AMT_INCOME_TOTAL']
application_test_new['CREDIT_TERM'] = application_test_new['AMT_ANNUITY'] / application_test_new['AMT_CREDIT']
application_test_new['DAYS_EMPLOYED_PERCENT'] = application_test_new['DAYS_EMPLOYED'] / application_test_new['DAYS_BIRTH']
application_test_new['GOOD_PRICE_PERCENT'] = application_test_new['AMT_GOODS_PRICE'] / application_test_new['AMT_INCOME_TOTAL']
application_test_new['CHILDREN_PERCENT'] = application_test_new['CNT_CHILDREN'] / application_test_new['CNT_FAM_MEMBERS']
application_test_new['OBS_30_CNT_SOCIAL_CIRCLE_REGION'] = application_test_new.OBS_30_CNT_SOCIAL_CIRCLE * application_test_new.REGION_POPULATION_RELATIVE
application_test_new['DEF_30_CNT_SOCIAL_CIRCLE_REGION'] = application_test_new.DEF_30_CNT_SOCIAL_CIRCLE * application_test_new.REGION_POPULATION_RELATIVE
application_test_new['OBS_60_CNT_SOCIAL_CIRCLE_REGION'] = application_test_new.OBS_60_CNT_SOCIAL_CIRCLE * application_test_new.REGION_POPULATION_RELATIVE
application_test_new['DEF_60_CNT_SOCIAL_CIRCLE_REGION'] = application_test_new.DEF_60_CNT_SOCIAL_CIRCLE * application_test_new.REGION_POPULATION_RELATIVE
application_train_new.replace([np.inf,-np.inf],np.nan,inplace=True)
application_train_new.isnull().sum().tail(11)
application_test_new.replace([np.inf,-np.inf],np.nan,inplace=True)
application_test_new.isnull().sum().tail(11)
a = application_train_new.columns.difference(application_test_new.columns)
a= a.drop('TARGET')

application_train_new.drop(a,inplace=True,axis=1)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
X = application_train_new.drop('TARGET',axis=1)
y = application_train_new.TARGET
X_scaled = scalar.fit_transform(X)

rf = RandomForestClassifier(n_jobs=-1,n_estimators=30)
rf.fit(X_scaled,y)
result_rf = pd.DataFrame()
result_rf['Features'] = X.columns
result_rf ['Values'] = rf.feature_importances_
result_rf.sort_values('Values',inplace=True, ascending = False)
result_rf.head(2)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_scaled,y)
a = lr.coef_[0]
coef = pd.Series(a, index=X.columns)
coef.sort_values(inplace=True)

coef.tail(2)
plt.figure(figsize = (6, 4))

# KDE plot of loans that were repaid on time
sns.kdeplot(application_train.loc[application_train['TARGET'] == 0, 'DAYS_ID_PUBLISH'], label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(application_train.loc[application_train['TARGET'] == 1, 'DAYS_ID_PUBLISH'], label = 'target == 1')

# Labeling of plot
plt.xlabel('DAYS_ID_PUBLISH'); plt.ylabel('Density'); plt.title('Distribution DAYS_ID_PUBLISH');
plt.figure(figsize = (6, 4))

# KDE plot of loans that were repaid on time
sns.kdeplot(application_train.loc[application_train['TARGET'] == 0, 'DAYS_LAST_PHONE_CHANGE'], label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(application_train.loc[application_train['TARGET'] == 1, 'DAYS_LAST_PHONE_CHANGE'], label = 'target == 1')

# Labeling of plot
plt.xlabel('DAYS_LAST_PHONE_CHANGE'); plt.ylabel('Density'); plt.title('Distribution DAYS_LAST_PHONE_CHANGE');
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import datetime

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
result_df = pd.DataFrame(data=np.ones(shape=(5,9)),columns=['Model','TruePositive','FalsePositive','FalseNegative','TrueNegative','Accuracy'
                                                         ,'Precision','Recall','Specifity'])

pca = PCA(n_components = 100)
pca_X_train = pd.DataFrame(pca.fit_transform(X_train))
pca_X_train.index = X_train.index

pca_X_test = pd.DataFrame(pca.transform(X_test))
pca_X_test.index = X_test.index

clf_1 = RidgeClassifier()
clf_1.fit(pca_X_train, y_train)
tn, fn,fp, tp = confusion_matrix(y_test,clf_1.predict(pca_X_test)).ravel()
result_df.loc[0,'Model'] = str(clf_1).split('(')[0]
result_df.loc[0,'TruePositive'] = tp
result_df.loc[0,'FalsePositive'] = fp
result_df.loc[0,'FalseNegative'] = fn
result_df.loc[0,'TrueNegative'] = tn
result_df.loc[0,'Accuracy'] = (tp+tn)/(tp+tn+fp+fn)
result_df.loc[0,'Precision'] = tp/(tp+fp)
result_df.loc[0,'Recall'] = tp / (tp+fn)
result_df.loc[0,'Specifity'] = tn / (tn+fp)
print(datetime.datetime.now())

clf_4 = LogisticRegression(C=0.05)
clf_4.fit(pca_X_train, y_train)
tn, fn,fp, tp = confusion_matrix(y_test,clf_4.predict(pca_X_test)).ravel()
result_df.loc[1,'Model'] = str(clf_4).split('(')[0]
result_df.loc[1,'TruePositive'] = tp
result_df.loc[1,'FalsePositive'] = fp
result_df.loc[1,'FalseNegative'] = fn
result_df.loc[1,'TrueNegative'] = tn
result_df.loc[1,'Accuracy'] = (tp+tn)/(tp+tn+fp+fn)
result_df.loc[1,'Precision'] = tp/(tp+fp)
result_df.loc[1,'Recall'] = tp / (tp+fn)
result_df.loc[1,'Specifity'] = tn / (tn+fp)
print(datetime.datetime.now())

clf_5 = RandomForestClassifier(n_jobs=-1)
clf_5.fit(pca_X_train,y_train)
tn, fn,fp, tp = confusion_matrix(y_test,clf_5.predict(pca_X_test)).ravel()
result_df.loc[2,'Model'] = str(clf_5).split('(')[0]
result_df.loc[2,'TruePositive'] = tp
result_df.loc[2,'FalsePositive'] = fp
result_df.loc[2,'FalseNegative'] = fn
result_df.loc[2,'TrueNegative'] = tn
result_df.loc[2,'Accuracy'] = (tp+tn)/(tp+tn+fp+fn)
result_df.loc[2,'Precision'] = tp/(tp+fp)
result_df.loc[2,'Recall'] = tp / (tp+fn)
result_df.loc[2,'Specifity'] = tn / (tn+fp)
print(datetime.datetime.now())

clf_6 = LinearDiscriminantAnalysis()
clf_6.fit(pca_X_train, y_train)
tn, fn,fp, tp= confusion_matrix(y_test,clf_6.predict(pca_X_test)).ravel()
result_df.loc[3,'Model'] = str(clf_6).split('(')[0]
result_df.loc[3,'TruePositive'] = tp
result_df.loc[3,'FalsePositive'] = fp
result_df.loc[3,'FalseNegative'] = fn
result_df.loc[3,'TrueNegative'] = tn
result_df.loc[3,'Accuracy'] = (tp+tn)/(tp+tn+fp+fn)
result_df.loc[3,'Precision'] = tp/(tp+fp)
result_df.loc[3,'Recall'] = tp / (tp+fn)
result_df.loc[3,'Specifity'] = tn / (tn+fp)
print(datetime.datetime.now())

clf_7 = QuadraticDiscriminantAnalysis()
clf_7.fit(pca_X_train,y_train)
tn, fn,fp, tp = confusion_matrix(y_test,clf_7.predict(pca_X_test)).ravel()
result_df.loc[4,'Model'] = str(clf_7).split('(')[0]
result_df.loc[4,'TruePositive'] = tp
result_df.loc[4,'FalsePositive'] = fp
result_df.loc[4,'FalseNegative'] = fn
result_df.loc[4,'TrueNegative'] = tn
result_df.loc[4,'Accuracy'] = (tp+tn)/(tp+tn+fp+fn)
result_df.loc[4,'Precision'] = tp/(tp+fp)
result_df.loc[4,'Recall'] = tp / (tp+fn)
result_df.loc[4,'Specifity'] = tn / (tn+fp)
print(datetime.datetime.now())
result_df
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
df7 = application_train_new.copy()
df7.drop('TARGET',inplace=True,axis=1)
df7['TARGET'] = application_train_new['TARGET']

values = df7[df7.TARGET==1].values
df8 = df7.copy()
# configure bootstrap
n_iterations = 10
n_size = int(len(df7[df7.TARGET==1]) * 0.001)
# run bootstrap
for i in range(n_iterations):
    # prepare train and test sets
    train = resample(values, n_samples=n_size)
    test = np.array([a for a in values if a.tolist() not in train.tolist()])
    test_df = pd.DataFrame(test, columns = df7.columns)
    # fit model
    model = DecisionTreeClassifier()
    model.fit(train[:,:-1], train[:,-1])
    # evaluate model
    predictions = model.predict(test[:,:-1])
    score = accuracy_score(test[:,-1], predictions)
    if(score >0.9):
        df8 = pd.concat([df8,test_df])
        df8.reset_index(drop=True, inplace = True)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import datetime

X_train, X_test, y_train, y_test = train_test_split(df8.iloc[:,:-1],df8.iloc[:,-1],test_size=0.2,random_state=42)
result_df_boostrap = pd.DataFrame(data=np.ones(shape=(5,9)),columns=['Model','TruePositive','FalsePositive','FalseNegative','TrueNegative','Accuracy'
                                                         ,'Precision','Recall','Specifity'])

scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

pca = PCA(n_components = 100)
pca_X_train = pd.DataFrame(pca.fit_transform(X_train_scaled))
pca_X_train.index = X_train.index

pca_X_test = pd.DataFrame(pca.transform(X_test_scaled))
pca_X_test.index = X_test.index

clf_1 = RidgeClassifier()
clf_1.fit(pca_X_train, y_train)
tn, fn,fp, tp = confusion_matrix(y_test,clf_1.predict(pca_X_test)).ravel()
result_df_boostrap.loc[0,'Model'] = str(clf_1).split('(')[0]
result_df_boostrap.loc[0,'TruePositive'] = tp
result_df_boostrap.loc[0,'FalsePositive'] = fp
result_df_boostrap.loc[0,'FalseNegative'] = fn
result_df_boostrap.loc[0,'TrueNegative'] = tn
result_df_boostrap.loc[0,'Accuracy'] = (tp+tn)/(tp+tn+fp+fn)
result_df_boostrap.loc[0,'Precision'] = tp/(tp+fp)
result_df_boostrap.loc[0,'Recall'] = tp / (tp+fn)
result_df_boostrap.loc[0,'Specifity'] = tn / (tn+fp)
print(datetime.datetime.now())


clf_4 = LogisticRegression(C=0.05)
clf_4.fit(pca_X_train, y_train)
tn, fn,fp, tp = confusion_matrix(y_test,clf_4.predict(pca_X_test)).ravel()
result_df_boostrap.loc[1,'Model'] = str(clf_4).split('(')[0]
result_df_boostrap.loc[1,'TruePositive'] = tp
result_df_boostrap.loc[1,'FalsePositive'] = fp
result_df_boostrap.loc[1,'FalseNegative'] = fn
result_df_boostrap.loc[1,'TrueNegative'] = tn
result_df_boostrap.loc[1,'Accuracy'] = (tp+tn)/(tp+tn+fp+fn)
result_df_boostrap.loc[1,'Precision'] = tp/(tp+fp)
result_df_boostrap.loc[1,'Recall'] = tp / (tp+fn)
result_df_boostrap.loc[1,'Specifity'] = tn / (tn+fp)
print(datetime.datetime.now())

clf_5 = RandomForestClassifier(n_jobs=-1)
clf_5.fit(pca_X_train,y_train)
tn, fn,fp, tp = confusion_matrix(y_test,clf_5.predict(pca_X_test)).ravel()
result_df_boostrap.loc[2,'Model'] = str(clf_5).split('(')[0]
result_df_boostrap.loc[2,'TruePositive'] = tp
result_df_boostrap.loc[2,'FalsePositive'] = fp
result_df_boostrap.loc[2,'FalseNegative'] = fn
result_df_boostrap.loc[2,'TrueNegative'] = tn
result_df_boostrap.loc[2,'Accuracy'] = (tp+tn)/(tp+tn+fp+fn)
result_df_boostrap.loc[2,'Precision'] = tp/(tp+fp)
result_df_boostrap.loc[2,'Recall'] = tp / (tp+fn)
result_df_boostrap.loc[2,'Specifity'] = tn / (tn+fp)
print(datetime.datetime.now())

clf_6 = LinearDiscriminantAnalysis()
clf_6.fit(pca_X_train, y_train)
tn, fn,fp, tp= confusion_matrix(y_test,clf_6.predict(pca_X_test)).ravel()
result_df_boostrap.loc[3,'Model'] = str(clf_6).split('(')[0]
result_df_boostrap.loc[3,'TruePositive'] = tp
result_df_boostrap.loc[3,'FalsePositive'] = fp
result_df_boostrap.loc[3,'FalseNegative'] = fn
result_df_boostrap.loc[3,'TrueNegative'] = tn
result_df_boostrap.loc[3,'Accuracy'] = (tp+tn)/(tp+tn+fp+fn)
result_df_boostrap.loc[3,'Precision'] = tp/(tp+fp)
result_df_boostrap.loc[3,'Recall'] = tp / (tp+fn)
result_df_boostrap.loc[3,'Specifity'] = tn / (tn+fp)
print(datetime.datetime.now())

clf_7 = QuadraticDiscriminantAnalysis()
clf_7.fit(pca_X_train,y_train)
tn, fn,fp, tp = confusion_matrix(y_test,clf_7.predict(pca_X_test)).ravel()
result_df_boostrap.loc[4,'Model'] = str(clf_7).split('(')[0]
result_df_boostrap.loc[4,'TruePositive'] = tp
result_df_boostrap.loc[4,'FalsePositive'] = fp
result_df_boostrap.loc[4,'FalseNegative'] = fn
result_df_boostrap.loc[4,'TrueNegative'] = tn
result_df_boostrap.loc[4,'Accuracy'] = (tp+tn)/(tp+tn+fp+fn)
result_df_boostrap.loc[4,'Precision'] = tp/(tp+fp)
result_df_boostrap.loc[4,'Recall'] = tp / (tp+fn)
result_df_boostrap.loc[4,'Specifity'] = tn / (tn+fp)
print(datetime.datetime.now())
result_df_boostrap
