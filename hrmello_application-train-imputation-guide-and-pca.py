import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np

app_train = pd.read_csv('../input/application_train.csv')
# application_test= pd.read_csv('../input/application_test.csv')
# bureau = pd.read_csv('../input/bureau.csv')
# bureau_balance = pd.read_csv('../input/bureau_balance.csv')
# POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
# credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
# previous_application = pd.read_csv('../input/previous_application.csv')
# installments_payments = pd.read_csv('../input/installments_payments.csv')
null_values_apptr = app_train.isnull().sum()
null_values_apptr = null_values_apptr[null_values_apptr != 0].sort_values(ascending = False).reset_index() #only show rows with null values
null_values_apptr.columns = ["variable", "n_missing"]
null_values_apptr.head()
for variable in null_values_apptr["variable"]:
    if (variable.endswith("MEDI")|variable.endswith("MODE")|variable.endswith("AVG")):
        app_train.loc[:,variable] = app_train.loc[:,variable].fillna(0)
for variable in null_values_apptr["variable"]:
    if (variable.startswith("AMT_REQ_CREDIT_BUREAU")):
        app_train.loc[:,variable] = app_train.loc[:,variable].fillna(0)
for variable in null_values_apptr["variable"]:
    if (variable.endswith("SOCIAL_CIRCLE")):
        app_train.loc[:,variable] = app_train.loc[:,variable].fillna(0)
#checking for remaining nulls:
null_values_apptr = app_train.isnull().sum()
null_values_apptr = null_values_apptr[null_values_apptr != 0].sort_values(ascending = False).reset_index() #only show rows with null values
null_values_apptr.columns = ["variable", "n_missing"]
#percentage of missing values on a given column
null_values_apptr["pct_missing"] = null_values_apptr.n_missing/len(app_train)
null_values_apptr
sns.kdeplot(app_train.OWN_CAR_AGE);
app_train["OWN_CAR"] = 0
app_train.loc[app_train.OWN_CAR_AGE >= 0, "OWN_CAR"] = 1
app_train.loc[:,("OWN_CAR", "OWN_CAR_AGE")].head()

#dropping car age column
app_train = app_train.drop(columns=["OWN_CAR_AGE"])
sns.kdeplot(app_train.EXT_SOURCE_1);
sns.kdeplot(app_train.EXT_SOURCE_2);
sns.kdeplot(app_train.EXT_SOURCE_3);
app_train = app_train.drop(columns= ["EXT_SOURCE_1"])
app_train = app_train.drop(columns= ["EXT_SOURCE_3"])
app_train.loc[:,"EXT_SOURCE_2"] = app_train.loc[:,"EXT_SOURCE_2"].fillna(app_train.EXT_SOURCE_2.median())
# app_train.loc[:,"EXT_SOURCE_3"] = app_train.loc[:,"EXT_SOURCE_3"].fillna(app_train.EXT_SOURCE_3.mode()[0])
null_values_apptr = app_train.isnull().sum()
null_values_apptr = null_values_apptr[null_values_apptr != 0].sort_values(ascending = False).reset_index() #only show rows with null values
null_values_apptr.columns = ["variable", "n_missing"]
null_values_apptr
sns.kdeplot(app_train.EXT_SOURCE_2);
# sns.kdeplot(app_train.EXT_SOURCE_3);
app_train.NAME_EDUCATION_TYPE.unique()
sns.heatmap(pd.crosstab(app_train.OCCUPATION_TYPE, app_train.NAME_EDUCATION_TYPE), cmap="Blues");
for education in app_train.NAME_EDUCATION_TYPE.unique():
    mode_to_impute = app_train[app_train.NAME_EDUCATION_TYPE == education].OCCUPATION_TYPE.mode()[0]
    app_train.loc[app_train.NAME_EDUCATION_TYPE == education, "OCCUPATION_TYPE"] = app_train.loc[app_train.NAME_EDUCATION_TYPE == education, "OCCUPATION_TYPE"].fillna(mode_to_impute)
null_values_apptr = app_train.isnull().sum()
null_values_apptr = null_values_apptr[null_values_apptr != 0].sort_values(ascending = False).reset_index() #only show rows with null values
null_values_apptr.columns = ["variable", "n_missing"]
null_values_apptr
app_train.NAME_TYPE_SUITE = app_train.NAME_TYPE_SUITE.astype("category")
sns.countplot(y = app_train.NAME_TYPE_SUITE)
app_train.NAME_TYPE_SUITE = app_train.NAME_TYPE_SUITE.fillna(app_train.NAME_TYPE_SUITE.mode()[0])

null_values_apptr = app_train.isnull().sum()
null_values_apptr = null_values_apptr[null_values_apptr != 0].sort_values(ascending = False).reset_index() #only show rows with null values
null_values_apptr.columns = ["variable", "n_missing"]
null_values_apptr
sns.countplot(app_train.CNT_FAM_MEMBERS)
app_train.CNT_FAM_MEMBERS = app_train.CNT_FAM_MEMBERS.fillna(app_train.CNT_FAM_MEMBERS.mode()[0])
sns.kdeplot(app_train.DAYS_LAST_PHONE_CHANGE)
app_train.DAYS_LAST_PHONE_CHANGE = app_train.DAYS_LAST_PHONE_CHANGE.fillna(app_train.DAYS_LAST_PHONE_CHANGE.mode()[0])
sns.kdeplot(app_train.AMT_ANNUITY)
app_train.AMT_ANNUITY = app_train.AMT_ANNUITY.fillna(app_train.AMT_ANNUITY.median())
sns.distplot(app_train.AMT_GOODS_PRICE[pd.notnull(app_train.AMT_GOODS_PRICE)])
app_train.AMT_GOODS_PRICE = app_train.AMT_GOODS_PRICE.fillna(app_train.AMT_GOODS_PRICE.mode()[0])
sns.distplot(app_train.AMT_GOODS_PRICE)
app_train.NAME_TYPE_SUITE = app_train.NAME_TYPE_SUITE.fillna(app_train.NAME_TYPE_SUITE.mode()[0])

null_values_apptr = app_train.isnull().sum()
null_values_apptr = null_values_apptr[null_values_apptr != 0].sort_values(ascending = False).reset_index() #only show rows with null values
null_values_apptr.columns = ["variable", "n_missing"]
null_values_apptr
######################################
############### PCA ##################
######################################

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
df = app_train.copy()
df = df.drop(columns = ["SK_ID_CURR", "TARGET"])
df = pd.get_dummies(df)

#scaling the components
ss = StandardScaler()
df = ss.fit_transform(df)

pca = PCA(random_state = 0)
df_decomposed = pca.fit(df)
df = pca.transform(df)
cumulative_var = []
cumul_var = 0
for exvar in df_decomposed.explained_variance_ratio_:
    cumul_var = exvar + cumul_var
    cumulative_var.append(cumul_var)
plt.plot(cumulative_var, label = "cumulative explained variance")
plt.plot(df_decomposed.explained_variance_ratio_, label = "individual explained variance")
plt.legend();