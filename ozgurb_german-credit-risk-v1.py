# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_credit = pd.read_csv("../input/germany-credit/german_credit_data.csv",index_col=0)
df_credit
df_credit["Credit_amount_log"]=np.log(df_credit["Credit amount"])

df_credit["Credit_amount_log"].hist()


t1 = df_credit.groupby(["Sex","Risk"])

t2 = df_credit.groupby(["Sex"])

t3 = t1[["Risk"]].count()/t2[["Risk"]].count()*100

t3
t1 = df_credit.groupby(["Job","Risk"])

t2 = df_credit.groupby(["Job"])

t3 = t1[["Risk"]].count()/t2[["Risk"]].count()*100

t3
t1 = df_credit.groupby(["Housing","Risk"])

t2 = df_credit.groupby(["Housing"])

t3 = t1[["Risk"]].count()/t2[["Risk"]].count()*100

t3
t1 = df_credit.groupby(["Checking account","Risk"])

t2 = df_credit.groupby(["Checking account"])

t3 = t1[["Risk"]].count()/t2[["Risk"]].count()*100

t3
def drawscatter(df_train,x,y):

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.scatter(x = df_train[x], y = df_train[y])

    plt.ylabel('y', fontsize=13)

    plt.xlabel('x', fontsize=13)

    plt.show()



def drawHist(df_train,x,y):

    df_train[x].hist(by=df_credit[y])

    
drawHist(df_credit,"Age","Risk")

drawHist(df_credit,"Sex","Risk")

drawHist(df_credit,"Job","Risk")

drawHist(df_credit,"Housing","Risk")

drawHist(df_credit,"Saving accounts","Risk")

drawHist(df_credit,"Purpose","Risk")
#drawHist(df_credit,"Credit amount","Risk")

#drawHist(df_credit,"Duration","Risk")



df_credit_2=df_credit
# check null values

def check_nulls(df_train):

    # check distinct values for the columns that contains null value

    # remove nan values that is total count less than 5 percent

    for d in df_train.columns:

        if  df_train[d].isnull().values.any():

            print("column "+d)

            if df_train[d].dtype.kind in 'bifc':

                df_train[d].fillna(0,inplace = True)

            else:

                print("column "+d)

                df_train[d].fillna("NULL_VALUE", inplace = True)

                

                

check_nulls(df_credit_2)



#check 

nan_cols = [i for i in df_credit_2.columns if df_credit_2[i].isnull().any()]

print(nan_cols)
# check extreme values for numeric inputs

for d in df_credit_2.columns:

    if df_credit_2[d].dtype.kind in 'bifc':

        print(d)

        print(df_credit_2[d].describe())
df_credit_2.rename(columns = {'Credit amount':'Credit_amount'}, inplace = True) 

df_credit_2.rename(columns = {'Saving accounts':'Saving_accounts'}, inplace = True) 

df_credit_2.rename(columns = {'Checking account':'Checking_account'}, inplace = True) 





# According to results above  focus on Credit amount

drawscatter(df_credit_2,"Credit_amount","Risk")

# bad: lets take values less than 15000

drawscatter(df_credit_2.query("(Credit_amount<14000 and Risk=='bad') or (Credit_amount<12600 and Risk=='good') "),"Credit_amount","Risk")





#check skewness

df_credit_2[["Credit_amount","Risk"]].query("(Credit_amount<14000 and Risk=='bad') or (Credit_amount<12600 and Risk=='good')").hist()

# try to make it more normally distributed

df_credit_2["Credit_amount_log"]=np.log(df_credit["Credit_amount"])

df_credit_2[["Credit_amount_log"]].hist()



#  for model Credit_amount_log will be used for predicition
df_credit_2.Age.unique()

bins = [0, 10, 18, 25, 30, 45, 60, 120]

labels = [1,2,3,4,5,6,7]

df_credit_2['Age_group'] = pd.cut(df_credit_2['Age'], bins=bins, labels=labels)



df_credit_2[["Age_group","Age"]]
df_credit_2.dtypes
for col in df_credit_2.columns:

    if df_credit_2[col].dtype.kind  in 'bifc':

        print("** "+col)



# correlation for  numeric and vs Risk ( categorical variable) 



def One_way_ANOVA(df_train):

    lister12 = []

    for col in df_train.columns:

        if df_train[col].dtype.kind  in 'bifc':

            import statsmodels.api as sm

            from statsmodels.formula.api import ols

            model = ols(col+' ~ Risk',data=df_train).fit()

            table = sm.stats.anova_lm(model, typ=2)

            #print(col)

            print(table["PR(>F)"][0])

            if table["PR(>F)"][0] < 0.05:

                lister12.append(col)

    return lister12



One_way_ANOVA(df_credit_2)
# correlation for  categorical values vs Risk ( categorical variable)  Chi-square Test of Independence

def cat_corr(df_train):

    lister12 = []

    for col in df_train.columns:

        if df_train[col].dtype.kind not in 'bifc':

            print(col)

            import pandas as pd

            confusion_matrix = pd.crosstab(df_train[col], df_train["Risk"])

            from scipy import stats

            print(stats.chi2_contingency(confusion_matrix))

    

            

cat_corr(df_credit_2)
pd.crosstab(df_credit_2["Purpose"], df_credit_2["Risk"])
pd.crosstab(df_credit_2["Checking_account"], df_credit_2["Risk"])
pd.crosstab(df_credit_2["Duration"], df_credit_2["Risk"]).apply(lambda r: r/r.sum(), axis=1)
# label encoder

def  labeler(df):

    from sklearn.preprocessing import LabelEncoder

    for col in df.columns:

        if df[col].dtype.kind not in 'bifc':

                lb_make = LabelEncoder()

                df[col] = lb_make.fit_transform(df[col])

                

labeler(df_credit_2)
df_credit_2
df_credit_2.Risk.unique()
from sklearn.preprocessing import OneHotEncoder



enc = OneHotEncoder(handle_unknown='ignore')



enc_df = pd.DataFrame(enc.fit_transform(df_credit_2[['Sex']]).toarray())

enc_df.columns = enc.get_feature_names(['Sex'])

# merge with main df bridge_df on key values

df_credit_2 = df_credit_2.join(enc_df)





enc_df = pd.DataFrame(enc.fit_transform(df_credit_2[['Housing']]).toarray())

enc_df.columns = enc.get_feature_names(['Housing'])

# merge with main df bridge_df on key values

df_credit_2 = df_credit_2.join(enc_df)



enc_df = pd.DataFrame(enc.fit_transform(df_credit_2[['Saving_accounts']]).toarray())

enc_df.columns = enc.get_feature_names(['Saving_accounts'])

# merge with main df bridge_df on key values

df_credit_2 = df_credit_2.join(enc_df)



enc_df = pd.DataFrame(enc.fit_transform(df_credit_2[['Checking_account']]).toarray())

enc_df.columns = enc.get_feature_names(['Checking_account'])

# merge with main df bridge_df on key values

df_credit_2 = df_credit_2.join(enc_df)









df_credit_2
df_credit_2.Duration.describe()
df_credit_2.columns
# gather cont. and categorical variable that have a strong relation with Risk



df_credit_3=df_credit_2[['Age', 'Duration', 'Credit_amount_log',       

        'Sex_0', 'Sex_1', 'Housing_0', 'Housing_1', 'Housing_2',

       'Saving_accounts_0', 'Saving_accounts_1', 'Saving_accounts_2',

       'Saving_accounts_3', 'Saving_accounts_4', 'Checking_account_0',

       'Checking_account_1', 'Checking_account_2', 'Checking_account_3',"Age_group"]]

df_credit_4=df_credit_2["Risk"]
# using cross-validation for splitting train data into two sets



from sklearn.model_selection import KFold # import KFold



cv = KFold(n_splits=5, random_state=42, shuffle=True)



for train_index, test_index in cv.split(df_credit_3):

    X_train, X_test, y_train, y_test = df_credit_3.iloc[train_index], df_credit_3.iloc[test_index], df_credit_4.iloc[train_index], df_credit_4.iloc[test_index]

    

    import xgboost as xgb

    from sklearn.metrics import mean_squared_error

    from math import sqrt

    from sklearn.metrics import r2_score

    

    params= {

    # Parameters that we are going to tune.

    'n_estimators':100,

    'max_depth':6, #Result of tuning with CV

    'eta':0.01

    }

    

    

    model = xgb.XGBClassifier()

    

    model.fit(X_train, y_train)

    xgb_pred = (model.predict(X_test))

    

    #print(xgb_pred)

    



    

    

    from sklearn.metrics import accuracy_score

    accuracy=round(accuracy_score( y_test , xgb_pred ) * 100, 2)            

    print(accuracy)

   