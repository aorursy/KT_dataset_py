# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
credit=pd.read_csv('../input/UCI_Credit_Card.csv')
# import the credit data set into python
credit.shape # number of observations and number of variables
list(credit.columns)
#checking the data types for each and every variable
credit.dtypes
dups=credit.duplicated()
dups.value_counts()
dups1=credit.duplicated(["ID"]) # checking the duplicates at id level
dups1.value_counts()
credit.head(4)
print(credit.isnull().sum())
credit.loc[credit["BILL_AMT1"].isnull() & credit["PAY_AMT1"].isnull()]
credit=credit.fillna(method="bfill")
# lets reconfirm is there any missing values in our data
print(credit.isnull().sum())
credit.head(4)
#checking the outliers for LIMIT_BAL variable
sns.boxplot(credit["LIMIT_BAL"])
from statsmodels.graphics.gofplots import qqplot
qqplot(credit["LIMIT_BAL"],line="s")
#checking the count of outliers
np.sum(np.where(credit["LIMIT_BAL"]>credit["LIMIT_BAL"].quantile(0.95),1,0))
np.sum(np.where(credit["LIMIT_BAL"]<credit["LIMIT_BAL"].quantile(0.025),1,0))
credit["LIMIT_BAL"].describe()
credit["LIMIT_BAL"].quantile([0,0.25,0.5,0.75,0.9,0.95,0.997,1])
# creating the user defined function for limit bins
def abc(LIMIT_BAL):
    if LIMIT_BAL>=10000.0 and LIMIT_BAL<=50000.0:
        return "lim_1"
    elif LIMIT_BAL>50000.0 and LIMIT_BAL<=140000.0:
        return "lim_2"
    elif LIMIT_BAL>140000.0 and LIMIT_BAL<=240000.0:
        return "lim_3"
    elif LIMIT_BAL>240000.0 and LIMIT_BAL<=430000.0:
        return "lim_4"
    else:
        return "lim_5"
credit["lim_bin"]=credit["LIMIT_BAL"].apply(abc)
credit["lim_bin"].value_counts()
# check number of levels
credit.nunique()
# drop the LIMIT_BAL variable
credit=credit.drop(["LIMIT_BAL"],axis=1)
credit.shape
sns.boxplot(credit["AGE"])
qqplot(credit["AGE"],line="s")
event_rate=(credit["default.payment.next.month"].value_counts()[1]/len(credit))*100
print(event_rate)
credit.groupby("default.payment.next.month")["BILL_AMT1"].mean()
pd.crosstab(credit["lim_bin"],credit["default.payment.next.month"])
credit.groupby("lim_bin")["PAY_AMT1"].mean()
credit.groupby("default.payment.next.month")["AGE"].mean()
pd.crosstab(credit["default.payment.next.month"],credit["SEX"])
pd.crosstab(credit["EDUCATION"],credit["default.payment.next.month"])
pd.crosstab(credit["MARRIAGE"],credit["default.payment.next.month"])
sns.countplot(credit["default.payment.next.month"])
list(credit.columns)
credit["SEX"]=credit["SEX"].astype("str")
credit.dtypes
credit["EDUCATION"].value_counts()
credit["EDUCATION"]=np.where(credit["EDUCATION"]>=5,5,credit["EDUCATION"])
credit["EDUCATION"].value_counts()
credit["EDUCATION"]=credit["EDUCATION"].astype("str")
credit["MARRIAGE"]=credit["MARRIAGE"].astype("str")
credit.columns
credit["bill_flag1"]=np.where(credit["BILL_AMT1"]==credit["BILL_AMT2"],1,0)
credit["bill_flag2"]=np.where(credit["BILL_AMT1"]>credit["BILL_AMT2"],1,0)
credit["bill_flag3"]=np.where(credit["BILL_AMT1"]<credit["BILL_AMT2"],1,0)
pd.crosstab(credit["bill_flag1"],credit["default.payment.next.month"])
credit[credit["default.payment.next.month"]==1].groupby("bill_flag1")["PAY_AMT1"].mean()
list(credit.columns)
credit["PAY_0"]=credit["PAY_0"].astype("str")
credit["PAY_2"]=credit["PAY_2"].astype("str")
credit["PAY_3"]=credit["PAY_3"].astype("str")
credit["PAY_4"]=credit["PAY_4"].astype("str")
credit["PAY_5"]=credit["PAY_5"].astype("str")
credit["PAY_6"]=credit["PAY_6"].astype("str")
credit.dtypes
from statsmodels.graphics.gofplots import qqplot
qqplot(credit["AGE"],line="s")
credit["AGE"].describe()
def abc1(AGE):
    if AGE>=21 and AGE<=28:
        return "age_bin1"
    elif AGE>28 and AGE<=34:
        return "age_bin2"
    elif AGE>34 and AGE<=41:
        return "age_bin3"
    else:
        return "age_bin4"
credit["AGE"]=credit["AGE"].apply(abc1)
credit["AGE"].value_counts()
pd.crosstab(credit["AGE"],credit["default.payment.next.month"])
credit1=credit.drop(["ID","bill_flag1","bill_flag2","bill_flag3"],axis=1)
credit1.shape
credit2=pd.get_dummies(credit1)
credit2.shape
credit2.head(4)
credit2["SEX_1"].dtypes
# import packages# import 
import pandas as pd
import numpy as np
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback
import string

max_bin = 20
force_bin = 5

# define a binning function
def mono_bin(Y, X, n = max_bin):
    
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1 
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n = force_bin         
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
        d2 = d1.groupby('Bucket', as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3=d3.reset_index(drop=True)
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.sum().EVENT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.EVENT_RATE/d3.NON_EVENT_RATE)
    d3["IV"] = (d3.EVENT_RATE-d3.NON_EVENT_RATE)*np.log(d3.EVENT_RATE/d3.NON_EVENT_RATE)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'WOE', 'IV']]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    
    return(d3)

def char_bin(Y, X):
        
    df1 = pd.DataFrame({"X": X, "Y": Y})
    df2 = df1.groupby('X',as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.groups
    d3["MAX_VALUE"] = df2.groups
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y     
    d3["EVENT_RATE"] = d3.EVENT/d3.sum().EVENT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.EVENT_RATE/d3.NON_EVENT_RATE)
    d3["IV"] = (d3.EVENT_RATE-d3.NON_EVENT_RATE)*np.log(d3.EVENT_RATE/d3.NON_EVENT_RATE)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'WOE', 'IV']]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    
    return(d3)

def data_vars(df1, target):
    
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]
    
    x = df1.dtypes.index
    count = -1
    
    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i            
                count = count + 1
                
            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv,ignore_index=True)
    
    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return(iv_df,iv)
IV = data_vars(credit2,credit2["default.payment.next.month"])
IV
IV[1]
IV1=IV[1]
IV1.head()
IV1["flag"]=np.where((IV1["IV"]>=0.02) & (IV1["IV"]<0.5),1,0)
IV1.head()
IV1["flag"].value_counts()
Iv2=IV1[IV1["flag"]==1]
len(Iv2)
credit3=credit2[Iv2["VAR_NAME"]]
credit3.shape
credit3["target"]=credit2["default.payment.next.month"]
credit3["target"].value_counts()
credit3.dtypes
import statsmodels.formula.api as sm
def vif_cal(input_data, dependent_col):
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.ols(formula="y~x", data=x_vars).fit().rsquared  
        vif=round(1/(1-rsq),2)
        print (xvar_names[i], " VIF = " , vif)
vif_cal(input_data=credit3, dependent_col="target")
# drop the PAY_2_0  because of VIF high
credit3=credit3.drop(["PAY_2_0"],axis=1)

credit3.shape
vif_cal(input_data=credit3, dependent_col="target")
credit3=credit3.drop(["PAY_4_0"],axis=1)
credit3.shape
vif_cal(input_data=credit3, dependent_col="target")
credit3=credit3.drop(["PAY_5_0"],axis=1)
credit3.shape
vif_cal(input_data=credit3, dependent_col="target")
credit3=credit3.drop(["PAY_3_0"],axis=1)
vif_cal(input_data=credit3, dependent_col="target")
credit3=credit3.drop(["PAY_0_0"],axis=1)
credit3.shape
vif_cal(input_data=credit3, dependent_col="target")
credit3=credit3.drop(["PAY_6_0"],axis=1)
credit3.shape
vif_cal(input_data=credit3, dependent_col="target")
credit3=credit3.drop(["PAY_2_-1"],axis=1)
credit3.shape
vif_cal(input_data=credit3, dependent_col="target")
credit3=credit3.drop(["PAY_4_-1"],axis=1)
credit3.shape
vif_cal(input_data=credit3, dependent_col="target")
credit3=credit3.drop(["PAY_5_2"],axis=1)
credit3.shape
vif_cal(input_data=credit3, dependent_col="target")
credit3=credit3.drop(["PAY_5_-1"],axis=1)
credit3.shape
vif_cal(input_data=credit3, dependent_col="target")
credit3=credit3.drop(["PAY_3_2"],axis=1)
credit3.shape
vif_cal(input_data=credit3, dependent_col="target")






