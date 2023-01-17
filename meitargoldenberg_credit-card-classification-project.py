# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# General tools
import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif,chi2
from itertools import product
from sklearn.model_selection import train_test_split as split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import math
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydot
from sklearn import metrics

# For transformations and predictions
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier,AdaBoostClassifier, GradientBoostingClassifier
# Imprt file to df
credit = pd.read_csv("/kaggle/input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv")
credit.head()
credit.info()
# Unique values per each feature
credit.nunique()
# Decribe of Data set
credit.describe(include='all').T
# renaming field PAY_0 to PAY_1
credit.rename(columns={"PAY_0": "PAY_1","default.payment.next.month":"default"},inplace=True)

# Aggregating Unknown educations type
education_dict = {2:3,3:2,4:0,5:0,6:0}
credit.EDUCATION.replace(education_dict,inplace=True)

# Adding field names descriptions
credit["SEX_desc"] = credit["SEX"].replace({1: "Male", 2: "Female"})
credit["EDUCATION_desc"] = credit["EDUCATION"].replace({0:"Other",1:"Graduate School", 2:"High School",3:"University"})
credit["MARRIAGE_desc"] = credit["MARRIAGE"].replace({1: "Married", 2: "Single",0:"Other",3:"Other"})

#----------------------------------------------------------
#???
credit["cum_pay"]=credit["PAY_AMT1"]+credit["PAY_AMT2"]+credit["PAY_AMT3"]+credit["PAY_AMT4"]+credit["PAY_AMT5"]
credit["cum_bill"]=(credit["BILL_AMT2"]+credit["BILL_AMT3"]+credit["BILL_AMT4"]+credit["BILL_AMT5"]+credit["BILL_AMT6"])

credit["percent_paid"]=(credit["cum_pay"]/credit["cum_bill"])
(credit["percent_paid"]).replace([np.inf, -np.inf], np.nan,inplace=True)
credit["percent_paid"].fillna(0,inplace=True)
credit["percent_paid"]=round(credit["percent_paid"],2)
#------------------------------------------------------------
# Concatenating all PAY Fields into 1 Field (3\6 months)
credit['con']=credit['PAY_1'].astype(str) +";" +credit['PAY_2'].astype(str)+";" +credit['PAY_3'].astype(str) +";" +credit['PAY_4'].astype(str)+";" +credit['PAY_5'].astype(str)+";" +credit['PAY_6'].astype(str) 
credit['con_3_months'] = credit['PAY_1'].astype(str) +";" +credit['PAY_2'].astype(str)+";" +credit['PAY_3'].astype(str)

delay = ['PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
credit["max_delay"]=credit[delay].max(axis=1)
# Recognizing Rolling credit of the 6 months using PAY_i = 0
roll_dict =  {'1':lambda row: int(row["PAY_1"] == 0),
              '2':lambda row: int(row["PAY_2"] == 0),
              '3':lambda row: int(row["PAY_3"] == 0),
              '4':lambda row: int(row["PAY_4"] == 0),
              '5':lambda row: int(row["PAY_5"] == 0),
              '6':lambda row: int(row["PAY_6"] == 0)}

for k,v in roll_dict.items():
   c = 'Roll_PAY_{}'.format(k)
   credit[c] = credit.apply(v,axis=1)

# Qnt of rolls over 3/6 months
credit['Total_Roll'] =  credit['Roll_PAY_1'] + credit['Roll_PAY_2'] + credit['Roll_PAY_3'] + credit['Roll_PAY_4'] + credit['Roll_PAY_5'] + credit['Roll_PAY_6']
credit['Three_Months_Roll'] =  credit['Roll_PAY_1'] + credit['Roll_PAY_2'] + credit['Roll_PAY_3']
#-------------------------------------------------------------
# Counting Amount of delays in 3\6 Months
Total_Delays = lambda row: int(row["PAY_1"]>0)+int(row["PAY_2"]>0)+int(row["PAY_3"]>0)+int(row["PAY_4"]>0)+int(row["PAY_5"]>0)+int(row["PAY_6"]>0)
Three_Months_Delays = lambda row: int(row["PAY_1"]>0)+int(row["PAY_2"]>0)+int(row["PAY_3"]>0)
                                                                         
credit['Total_Delays'] = credit.apply(Total_Delays,axis=1)
credit['Three_Months_Delays'] = credit.apply(Three_Months_Delays,axis=1)
#-------------------------------------------------------------
# Calculating the Trends
# we calculate the trend in the status, every month we calculate the status today minus the best ever for the client (the min is the best)
dict={'1':lambda row: min(int(row["PAY_2"]),int(row["PAY_3"]),int(row["PAY_4"]),int(row["PAY_5"]),int(row["PAY_6"])),
      '2':lambda row: min(int(row["PAY_3"]),int(row["PAY_4"]),int(row["PAY_5"]),int(row["PAY_6"])),
      '3':lambda row: min(int(row["PAY_4"]),int(row["PAY_5"]),int(row["PAY_6"])),
      '4':lambda row: min(int(row["PAY_5"]),int(row["PAY_6"])),
      '5':lambda row: int(row["PAY_6"])}

for k,v in dict.items():
   b = 'trend_{}'.format(k) #minus is a good trend
   c = 'PAY_{}'.format(k)
   credit['max_Step1'] = credit.apply(v,axis=1)
   credit['max_Step2'] = credit['max_Step1'].replace(-2,-1)
   credit[b] = credit[c].replace(-2,-1) - credit['max_Step2']
  
#credit.drop(['max_Step1','max_Step2'],axis=1,inplace=True)

# The mean of the trends for 3\6 Months
credit['mean_trend'] = credit[['trend_1','trend_2','trend_3','trend_4','trend_5']].mean(axis=1)
credit['mean_trend_3_months'] = credit[['trend_1','trend_2']].mean(axis=1)
#--------------------------------------------------------------
# Deviations of Last 3/6 months
# (All bills of [n months]) / Limit balance* [n months]

Deviation = lambda row: (int(row["BILL_AMT1"]) + int(row["BILL_AMT2"]) + int(row["BILL_AMT3"]) + int(row["BILL_AMT4"]) + int(row["BILL_AMT5"]) + int(row["BILL_AMT6"])) / (int(row["LIMIT_BAL"]) * 6)
Deviation_3_months = lambda row: (int(row["BILL_AMT1"]) + int(row["BILL_AMT2"]) + int(row["BILL_AMT3"])) / (int(row["LIMIT_BAL"]) * 3)

credit['Deviation'] = credit.apply(Deviation,axis=1)
credit['Deviation_3_months'] = credit.apply(Deviation_3_months,axis=1)
#----------------------------------------------------------------
# 5 fields of ratio between the payment and the bill (How much we paid from bill).
for i in range(1,6):
  a,b,c = 'Pay_from_Bill_{}'.format(i) , 'PAY_AMT{}'.format(i) , 'BILL_AMT{}'.format(i+1)
  credit[a] = credit[b] / credit[c]
  credit[a].replace([np.inf, -np.inf], np.nan,inplace=True)
  credit[a].fillna(0,inplace=True)

credit['mean_pay_from_Bill'] = credit[['Pay_from_Bill_1','Pay_from_Bill_2','Pay_from_Bill_3','Pay_from_Bill_4','Pay_from_Bill_5']].mean(axis=1)
#------------------------------------------------------------------
# Counting How many times paid less from the bill for (3\6 months)
count_out_of_bill = lambda row: int(row["Pay_from_Bill_1"]<1)+int(row["Pay_from_Bill_2"]<1)+int(row["Pay_from_Bill_3"]<1)+int(row["Pay_from_Bill_4"]<1)+int(row["Pay_from_Bill_5"]<1)
count_out_of_bill_3_months = lambda row: int(row["Pay_from_Bill_1"]<1)+int(row["Pay_from_Bill_2"]<1)+int(row["Pay_from_Bill_3"]<1)
                                                                         
credit['count_out_of_bill'] = credit.apply(count_out_of_bill,axis=1)
credit['count_out_of_bill_3_months'] = credit.apply(count_out_of_bill_3_months,axis=1)
#-------------------------------------------------------------------
# Standard deviation pay to bill - std(all payments) / std(all bills)
credit['std_pay_to_bill_temp1'] = credit[["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5"]].std(axis=1)
credit['std_pay_to_bill_temp2'] =  credit[["BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]].std(axis=1)

def value_0(row):
  if row.loc['std_pay_to_bill_temp2'] ==0:
    return 0
  else:
    return row.loc['std_pay_to_bill_temp1'] / row.loc['std_pay_to_bill_temp2']
  
credit['std_pay_to_bill'] = credit.apply(value_0,axis=1)

credit.drop(['std_pay_to_bill_temp1','std_pay_to_bill_temp2'],axis=1,inplace=True)
#-------------------------------------------------------------------
# trend Pay_from_Bill_1         [Pay_from_Bill_1 / (avg of all other pay from bill)]
credit['trend_pay_from_bill_1'] = credit[["Pay_from_Bill_2","Pay_from_Bill_3","Pay_from_Bill_4","Pay_from_Bill_5"]].mean(axis=1)
#--------------------------------------------------------------------
# quantile trend Pay_from_Bill_1         [Pay_from_Bill_1 / (quantile of all other pay from bill)]
credit["quantile_pay_from_bill_1"] = credit["Pay_from_Bill_1"] / credit[["Pay_from_Bill_2","Pay_from_Bill_3","Pay_from_Bill_4","Pay_from_Bill_5"]].quantile(axis=1)
#--------------------------------------------------------------------
# PAY_AMT6 / mean(PAY_AMT1-6)
credit['mean_pay_temp'] = credit[["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5"]].mean(axis=1)
credit['pay6_mean_1_to_6'] = credit['PAY_AMT6'] / credit['mean_pay_temp']

credit.drop(['mean_pay_temp'],axis=1,inplace=True)
#---------------------------------------------------------------------
# std(PAY_AMT1-5) / mean(PAY_AMT1-6)
credit['stdev_pay_temp'] = credit[["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5"]].std(axis=1)
credit['avg_pay_temp'] = credit[["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]].mean(axis=1)
credit['std_to_mean_pays'] = credit['stdev_pay_temp'] / credit['avg_pay_temp']

credit.drop(['stdev_pay_temp','avg_pay_temp'],axis=1,inplace=True)
#-----------------------------------------------------------------------
# stdev(BILL_AMT_1-6) / mean(BILL_AMT_1-6)
credit['stdev_bill_temp'] = credit[["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]].std(axis=1)
credit['mean_bill_temp'] = credit[["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]].mean(axis=1)
credit['std_to_mean_bill'] = credit['stdev_bill_temp'] / credit['mean_bill_temp'] # change the name to the opposite

credit.drop(['stdev_bill_temp','mean_bill_temp'],axis=1,inplace=True)
#------------------------------------------------------------------------
# replacing nan\inf values with zero
credit.replace([np.inf, -np.inf], np.nan,inplace=True)
credit.fillna(0,inplace=True)
#------------------------------------------------------------------------
# Changing education values
education_dict = {2:3,3:2,4:0,5:0,6:0}
credit.EDUCATION.replace(education_dict,inplace=True)
# No bills for all 6 months
no_bill = credit[(credit.BILL_AMT1==0)&(credit.BILL_AMT2==0) & (credit.BILL_AMT3==0)&(credit.BILL_AMT4==0)&(credit.BILL_AMT5==0)&(credit.BILL_AMT6==0)]
no_bill['group'] = 0

def no_bill(row):
  if ((row['BILL_AMT1']==0)&(row['BILL_AMT2']==0)&(row['BILL_AMT3']==0)&(row['BILL_AMT4']==0)&(row['BILL_AMT5']==0)&(row['BILL_AMT6']==0)):
    return 1
  else:
    return 0
credit['no_bill_flag'] = credit.apply(no_bill, axis=1)
credit.no_bill_flag.value_counts()  
#----------------------------------------------------------
# Total bills is lower then zero
credit['Total_Bills_LIMIT_BAL'] = (credit.BILL_AMT1 + credit.BILL_AMT2 + credit.BILL_AMT3 + credit.BILL_AMT4 + credit.BILL_AMT5 + credit.BILL_AMT6) / (credit.LIMIT_BAL)
credit['group'] = 0
test_2 = credit[credit.Total_Bills_LIMIT_BAL < 0]

def Total_bills(row):
    if row['Total_Bills_LIMIT_BAL'] < 0:
      return 1
    else:
      return 0

credit['Total_Bills_LIMIT_BAL_flag'] = credit.apply(Total_bills, axis=1)

credit.drop(['Total_Bills_LIMIT_BAL','group'],axis=1,inplace=True)
#---------------------------------------------------------
# Total Pay \ Total Bills is lower than 0\higher than 1
# credit['Total_of_Total'] = (credit.cum_pay)/(credit.cum_bill)
credit['group'] = 0
test_3 = credit[(credit.percent_paid < 0)]
test_3.groupby('group')['default'].sum() / test_3.groupby('group')['default'].count()

def Total_of_Total_zero(row):
    if row['percent_paid'] < 0:
      return 1
    else:
      return 0

credit['Total_of_Total_zero_flag'] = credit.apply(Total_of_Total_zero, axis=1)

#-----------------------------------------------------------------------------------------------------

credit['group'] = 0
test_4 = credit[(credit.percent_paid > 1) ]
test_4=test_4.groupby('group')['default'].sum() / test_4.groupby('group')['default'].count()

def Total_of_Total_one(row):
    if row['percent_paid'] > 1:
      return 1
    else:
      return 0

credit['Total_of_Total_one_flag'] = credit.apply(Total_of_Total_zero, axis=1)
credit.drop(['group'],axis=1,inplace=True)
# Creating Account rate using the 6 PAY fields and by metric

# 1. Converting PAY values into sutible rank
dict_convert={0:1,-2:1.3,-1:1.4,1:2.5,2:3,3:6,4:6,5:6,6:6,7:6,8:6}

credit['PAY_LM'] = credit['PAY_1']
credit.PAY_LM.replace(dict_convert,inplace=True)

for i in range(1,7):
  def metric_prep(row):
    a = {v for k, v in dict_convert.items() if row.loc['PAY_{}'.format(i)] == k}
    a = str(a)
    return float(a.replace('{','').replace('}',''))
  credit['metric_prep_{}'.format(i)] = credit.apply(metric_prep,axis=1)
  
# 2. Creating unique data frame with all combinations of PAY fields
unique_values = credit[['metric_prep_1','metric_prep_2','metric_prep_3','metric_prep_4','metric_prep_5','metric_prep_6']].drop_duplicates()
unique_values['Rate_key'] =  unique_values['metric_prep_1'].astype(str) +";" +unique_values['metric_prep_2'].astype(str) +";" +unique_values['metric_prep_3'].astype(str) +";" +unique_values['metric_prep_4'].astype(str) +";" +unique_values['metric_prep_5'].astype(str) +";" +unique_values['metric_prep_6'].astype(str)
credit['Rate_key']=credit['metric_prep_1'].astype(str) +";" +credit['metric_prep_2'].astype(str)+";" +credit['metric_prep_3'].astype(str) +";" +credit['metric_prep_4'].astype(str)+";" +credit['metric_prep_5'].astype(str)+";" +credit['metric_prep_6'].astype(str) 

# 3. Taking the main combinations (with most of the data) and use them as comparison group
rate1 = [1,1,1,1,1,1]             #  0 0 0 0 0 0 10.4%  9821 obs
rate2 = [1.3,1.3,1.3,1.3,1.3,1.3] # -2-2-2-2-2-2 13.4%  2109 obs
rate3 = [1.4,1.4,1.4,1.4,1.4,1.4] # -1-1-1-1-1-1 14.2%  1992 obs
rate4 = [2.5,1.3,1.3,1.3,1.3,1.3] # 1-2-2-2-2-2  36%     651 obs
rate5 = [5,5,5,5,5,5]             # 2 2 2 2 2 2  77.5%   530 obs

# 4. Using euclidean distance metric + weights over months
weights = [6/21,5/21,4/21,3/21,2/21,1/21]

def euclidean_distance(pt1,pt2):
  distance = 0
  for i,j in zip(range(len(pt1)),weights) :
    distance += j * (pt1[i] - pt2[i]) ** 2
  return distance ** 0.5  
  return distance

# 5. Converting unique_values df into list and than running each row and comparing it combination to the 5 ratings.
# We'll take the closest rate by using min func (the min distnace) and than tag the row to the closest rate
s=[]
for row in unique_values.values.tolist():
  Rate_1 = (euclidean_distance(row,rate1))
  Rate_2 = (euclidean_distance(row,rate2))
  Rate_3 = (euclidean_distance(row,rate3))
  Rate_4 = (euclidean_distance(row,rate4))
  Rate_5 = (euclidean_distance(row,rate5))

  dict={'Rate_1':Rate_1,'Rate_2':Rate_2,'Rate_3':Rate_3,'Rate_4':Rate_4,'Rate_5':Rate_5}
  s.append(list(dict.keys())[list(dict.values()).index(min(dict.values()))])

# 6. Merge between data frames
# 6.1 Converting data from list into df (and this how it looks like)
Rating_df = pd.DataFrame(data=s, index=None, columns=None, dtype=None, copy=False)
Rating_df.head()
# 6.2 Join between the tag (Rate_i) to it's relevant key (using index as the key)
unique_values.reset_index(inplace=True)
full_Rating_df = unique_values.join(Rating_df)

full_Rating_df.drop(['index','metric_prep_1','metric_prep_2','metric_prep_3','metric_prep_4','metric_prep_5','metric_prep_6'],axis=1,inplace=True)
full_Rating_df.rename(columns={0:"score_rate"},inplace=True)
# And this is how it looks after the join
full_Rating_df.head()
# 6.3 Join between the main df (credit) and the tagging data frame (full_Rating_df)
credit_rank = credit.merge(full_Rating_df,left_on='Rate_key',right_on='Rate_key',how='left')
credit_rank.head()
# 7 Validating the new tagging field
# 7.1 Ratio for default
credit_rank.groupby('score_rate')['default'].sum() / credit_rank.groupby('score_rate')['default'].count()

#7.2 Values Amount
example=credit_rank[(credit_rank.score_rate=='Rate_5')]
example.con.value_counts()
# Rounding the values for ration features
credit_rank["Deviation2"]=round(credit_rank["Deviation"],3)
credit_rank["AGE1"]=round(credit_rank["AGE"]/10,0)
credit_rank["AGE_round"]=round(credit_rank["AGE"]/5,0)
credit_rank['Deviation_round']=round(credit_rank['Deviation2']*10)
credit_rank['Deviation_round'].value_counts()
credit_rank['Pay_from_Bill_1_round']=round(credit_rank['Pay_from_Bill_1']*10) 
credit_rank['Pay_from_Bill_1_round'].value_counts()
credit_rank['mean_pay_from_Bill_round']=round(credit_rank['mean_pay_from_Bill']*10) 
credit_rank['mean_pay_from_Bill_round'].value_counts() 
credit_rank['trend_pay_from_bill_round']=round(credit_rank['trend_pay_from_bill_1']*10) 
credit_rank['trend_pay_from_bill_round'].value_counts().head(20) 

credit_rank["dev*limit_bal_"]=(credit_rank["Deviation2"]*credit_rank["LIMIT_BAL"])
# Dropping unnecesary fields
credit_rank.drop(['cum_pay','cum_bill','AGE','BILL_AMT1','max_Step1','max_Step2','mean_trend_3_months','Deviation','Deviation_3_months','Pay_from_Bill_1',
                  'mean_pay_from_Bill','std_pay_to_bill','trend_pay_from_bill_1','quantile_pay_from_bill_1','pay6_mean_1_to_6','std_to_mean_pays','std_to_mean_bill',
                  'Deviation2','AGE1'],axis=1,inplace=True)

# Dropping unnecesary fields
credit_rank.drop(['ID','PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
                  'PAY_AMT1','PAY_AMT2','PAY_AMT5','PAY_AMT3','PAY_AMT4','PAY_AMT6','SEX','EDUCATION','MARRIAGE','con','con_3_months',
                  'Roll_PAY_2','Roll_PAY_3','Roll_PAY_4','Roll_PAY_5','Roll_PAY_6',
                  'trend_2','trend_3','trend_4','trend_5',
                  'Pay_from_Bill_2','Pay_from_Bill_3','Pay_from_Bill_4','Pay_from_Bill_5',
                  'metric_prep_1','metric_prep_2','metric_prep_3','metric_prep_4','metric_prep_5','metric_prep_6','Rate_key',
                  ],axis=1,inplace=True)
#--------------------------------------------------------------------------------
# Dummies
# Creating Dummies for logistic regression \ KNN
credit_rank_1 = credit_rank.copy() # for decision tree 

SEX_field = pd.get_dummies(credit_rank.SEX_desc,prefix='SEX').iloc[:,1:]
credit_rank = pd.concat([credit_rank,SEX_field],axis=1)

EDUCATION_field = pd.get_dummies(credit_rank.EDUCATION_desc,prefix='EDUCATION').iloc[:,1:]
credit_rank = pd.concat([credit_rank,EDUCATION_field],axis=1)

MARRIAGE_field = pd.get_dummies(credit_rank.MARRIAGE_desc,prefix='MARRIAGE').iloc[:,1:]
credit_rank = pd.concat([credit_rank,MARRIAGE_field],axis=1)

score_rate_field = pd.get_dummies(credit_rank.score_rate,prefix='score_rate').iloc[:,1:]
credit_rank = pd.concat([credit_rank,score_rate_field],axis=1)


# for decision tree / random forest
SEX_field = pd.get_dummies(credit_rank_1.SEX_desc,prefix='SEX')
credit_rank_1 = pd.concat([credit_rank_1,SEX_field],axis=1)

EDUCATION_field = pd.get_dummies(credit_rank_1.EDUCATION_desc,prefix='EDUCATION')
credit_rank_1 = pd.concat([credit_rank_1,EDUCATION_field],axis=1)

MARRIAGE_field = pd.get_dummies(credit_rank_1.MARRIAGE_desc,prefix='MARRIAGE')
credit_rank_1 = pd.concat([credit_rank_1,MARRIAGE_field],axis=1)

score_rate_field = pd.get_dummies(credit_rank_1.score_rate,prefix='score_rate')
credit_rank_1 = pd.concat([credit_rank_1,score_rate_field],axis=1)

credit_rank_1.drop(['SEX_desc','EDUCATION_desc','MARRIAGE_desc','score_rate'],axis=1,inplace=True)
credit_rank.drop(['SEX_desc','EDUCATION_desc','MARRIAGE_desc','score_rate'],axis=1,inplace=True)
df_1 = credit[credit['default'] == 1]
df_0 = credit[credit['default'] == 0]
x_field = 'LIMIT_BAL'
title = 'LIMIT_BAL Distribution'
#plt.xlim(0, 10) 
ax = sns.distplot(round(df_0[x_field],0), kde=False, label = 'No Default')
ax = sns.distplot(round(df_1[x_field],0), kde=False, label = 'Default')
ax.set_title(title)
plt.legend()
df_1 = credit[credit['default'] == 1]
df_0 = credit[credit['default'] == 0]
x_field = 'max_delay'
title = 'max_delay Distribution'
#plt.xlim(-1, 2) 
ax = sns.distplot(round(df_0[x_field],0), kde=False, label = 'No Default')
ax = sns.distplot(round(df_1[x_field],0), kde=False, label = 'Default')
ax.set_title(title)
plt.legend()
credit["limit_1000"]=credit["LIMIT_BAL"]/1000
(credit.groupby('limit_1000')['default'].sum() / credit.groupby('limit_1000')['default'].count()).plot()
credit.groupby('PAY_1')['default'].count().plot()
# Histogram
credit[['AGE','EDUCATION','MARRIAGE','LIMIT_BAL',
        'BILL_AMT1','BILL_AMT2','BILL_AMT3',
        'BILL_AMT4','BILL_AMT5','PAY_1','PAY_2','PAY_3',
        'PAY_4','PAY_5','PAY_6']].hist(figsize=(30, 30),color=("c"))
# #???

# # Pairplot graph
# # I checked the default 0/1 and changed 0-->1 1--->0
# # can we use delete to drop credit_3 at the end?


# credit["def_desc"] = credit["default"].replace({0: "no-def", 1: "def"})

# a = sns.pairplot(credit[['LIMIT_BAL','PAY_1','BILL_AMT1','PAY_AMT1','def_desc',
#                        'Total_Roll','Total_Delays','mean_trend','Deviation']],hue='def_desc')
                       
# a.fig.set_size_inches(22,22)
# Comparing defaults classes by main metrics
print(credit.groupby('default')[['Total_Delays','Three_Months_Delays',
                                 'Total_Roll','Three_Months_Roll',
                                 'mean_trend','mean_trend_3_months',
                                 'Deviation','Deviation_3_months',
                                 'mean_pay_from_Bill',
                                 'count_out_of_bill','count_out_of_bill_3_months']].agg(['mean','min','max']).T)
# # Comparing default classes by mean_pay_from_Bill & Deviation (size = mean_trend)
# # only score_rate_Rate_4 group
# try1=credit_rank_1[credit_rank_1['score_rate_Rate_4']==1]

# fig = plt.figure(figsize=(20, 8))#, dpi=80)
# ax = fig.gca()
# plt.style.use('seaborn')
# ss1=[(s+1)*100 for s in try1['mean_trend']]
# cs= try1['default'].replace(1,'red').replace(0,'yellow')
# scatter=ax.scatter(x='Deviation', y='mean_pay_from_Bill',data=try1,s=ss1,c=cs,alpha=0.8,edgecolor='black',linewidth=1)

# handles, labels=scatter.legend_elements(prop='sizes')
# labels = sorted(try1[try1.mean_trend > 0]['mean_trend'].unique())

# plt.legend(handles,labels,title='mean_trend')

# ax.set_ylim((0,2))
# ax.set_xlim((0,0.5))
# ax.set_xlabel('Deviation')
# ax.set_ylabel('mean_pay_from_Bill')
# ax.set_title('Default_by {} and {}'.format('Deviation','mean_pay_from_Bill'))
# plt.show()
# Comparing default classes by mean_pay_from_Bill & Deviation (size = mean_trend)
fig = plt.figure(figsize=(20, 8))#, dpi=80)
ax = fig.gca()
plt.style.use('seaborn')
ss1=[(s+1)*100 for s in credit['mean_trend']]
cs= credit['default'].replace(1,'red').replace(0,'yellow')
scatter=ax.scatter(x='Deviation', y='mean_pay_from_Bill',data=credit,s=ss1,c=cs,alpha=0.8,edgecolor='black',linewidth=1)

handles, labels=scatter.legend_elements(prop='sizes')
labels = sorted(credit[credit.mean_trend > 0]['mean_trend'].unique())

plt.legend(handles,labels,title='mean_trend')

ax.set_ylim((-1,5))
ax.set_xlim((-0.05,2))
ax.set_xlabel('Deviation')
ax.set_ylabel('mean_pay_from_Bill')
ax.set_title('Default_by {} and {}'.format('Deviation','mean_pay_from_Bill'))
plt.show()
# Comparing default classes by PAY_AMT1 & Deviation (size = mean_trend)

fig = plt.figure(figsize=(20, 8))#, dpi=80)
ax = fig.gca()
plt.style.use('seaborn')
credit_p=credit[(credit.PAY_AMT1<350000)&(credit.Deviation<3)]
ss1=[(s+1)*100 for s in credit_p['mean_trend']]
cs= credit_p['default'].replace(1,'red').replace(0,'yellow')

scatter=ax.scatter(x='Deviation', y='PAY_AMT1',data=credit_p,s=ss1,c=cs,alpha=0.8,edgecolor='black',linewidth=1)

handles, labels=scatter.legend_elements(prop='sizes')
labels = sorted(credit_p[credit_p.mean_trend > 0]['mean_trend'].unique())

plt.legend(handles,labels,title='mean_trend')

ax.set_xlabel('Deviation')
ax.set_ylabel('PAY_AMT1')
ax.set_title('Default_by {} and {}'.format('Deviation','PAY_AMT1'))
plt.show()
# Comparing default classes by mean_trend & Deviation (size = Total_Roll)
fig = plt.figure(figsize=(20, 8))#, dpi=80)
ax = fig.gca()
plt.style.use('seaborn')
ss = (credit['Total_Roll']+1)*100
cs= credit['default'].replace(1,'red').replace(0,'yellow')

scatter=ax.scatter(x='Deviation', y='mean_trend',data=credit,s=ss,c=cs,alpha=0.8,edgecolor='black',linewidth=1)
handles, labels=scatter.legend_elements(prop='sizes')
labels=['0','1','2','3','4','5','6']
plt.legend(handles,labels,title='Total_Roll')

#ax.set_ylim((-1,2))
ax.set_xlim((-0.05,3))
ax.set_xlabel('Deviation')
ax.set_ylabel('mean_trend')
ax.set_title('Default_by {} and {}'.format('Deviation','mean_trend'))
plt.show()
# Default total rolls by default (1 or 0)
credit.hist(column='Total_Roll',by='default',color='c',sharex=True,sharey=True,figsize=(10,5),bins=7,histtype='bar',density='True')
plt.xlabel('Total_Roll')
plt.suptitle('default_Total_Roll', x=1, y=1.1, ha='center', fontsize='xx-large')
# default deviation by default (1 or 0)
credit.hist(column='mean_trend_3_months',by='default',bins=10,color='magenta',sharex=True,sharey=True,figsize=(10,5),histtype='bar')
plt.suptitle('default_mean_trend_3_months', x=.5, y=1.1, ha='center', fontsize='xx-large')
# Plot - % of Default zero and Total Qnt by Education
education_percent=credit[(credit.EDUCATION_desc == "Graduate School") | (credit.EDUCATION_desc == "High School") |  (credit.EDUCATION_desc == "University")].groupby('EDUCATION_desc')['default'].sum()/ credit[(credit.EDUCATION_desc == "Graduate School") | (credit.EDUCATION_desc == "High School") |  (credit.EDUCATION_desc == "University")].groupby('EDUCATION_desc')['default'].count()

fig, ax1 = plt.subplots(figsize=(15, 5))

education_percent.plot(secondary_y=True, marker='d')
credit[(credit.EDUCATION_desc == "Graduate School") | (credit.EDUCATION_desc == "High School") |  (credit.EDUCATION_desc == "University")].groupby('EDUCATION_desc')['default'].count().plot(kind='bar',color="pink")

plt.title("% of Default zeo by EDUCATION (Qnt and mean)")
plt.figure(figsize=(15,4))
# Plot - % of Default zero and Total Qnt by MARRIAGE
credit_1 = credit[(credit.MARRIAGE_desc != "Other") & (credit.MARRIAGE_desc != "0")]
education_percent=credit_1.groupby('MARRIAGE_desc')['default'].sum() / credit_1.groupby('MARRIAGE_desc')['default'].count()

fig, ax1 = plt.subplots(figsize=(15, 5))

education_percent.plot(secondary_y=True, marker='d')
credit_1.groupby('MARRIAGE_desc')['default'].count().plot(kind='bar',color="pink")

plt.title("% of Default zeo by MARRIAGE (Qnt and mean)")
plt.figure(figsize=(15,4))
# Plot - % of Default zero and Total Qnt by Gender
education_percent=credit.groupby('SEX_desc')['default'].sum() / credit.groupby('SEX_desc')['default'].count()

fig, ax1 = plt.subplots(figsize=(15, 5))

education_percent.plot(secondary_y=True, marker='d')
credit.groupby('SEX_desc')['default'].count().plot(kind='bar',color="pink")

plt.title("% of Default zeo by GENDER (Qnt and mean)")
plt.figure(figsize=(15,4))
# Calculate correlations for the prediction field:default
corr = credit.corr()
corr['default'].sort_values(ascending=False)
# Correlations by all fields
sns.set(style="ticks") #white, dark, whitegrid, darkgrid, ticks
f, ax = plt.subplots(figsize=(25, 25))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, s=80, l=55, n=9,as_cmap=True)
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
delays_pay1_percent = credit.groupby(['Total_Delays','count_out_of_bill'])['default'].sum() / credit.groupby(['Total_Delays','count_out_of_bill'])['default'].count()
delays_pay1_percent = delays_pay1_percent.to_frame()
delays_pay1_percent.rename(columns={"default":"Prob_default_1"},inplace=True)
delays_pay1_percent.reset_index(level=[0,1], inplace=True)

sns.heatmap(pd.crosstab(delays_pay1_percent.Total_Delays, delays_pay1_percent.count_out_of_bill, values=delays_pay1_percent.Prob_default_1, aggfunc='mean'),cmap="BuPu", annot=True, cbar=False)

plt.title('Probability for default 1 by Total_Delays & count_out_of_bill',x=0.5, y=0.7, ha='center',fontsize='xx-large')
#delays_pay1_percent
# Splitting data to X and y
X = credit_rank_1.drop('default', axis=1)
y = credit_rank_1['default']
num_X = X.select_dtypes(include=[np.number])

# We chose variance of 5% as indicator
selector = VarianceThreshold(0.02)
selector.fit_transform(num_X)

# The new df without dropped the fields
new_columns = num_X.columns[selector.get_support()]
new_num_X = num_X[new_columns]

# Suggested fields to remove
print("These are the fields the VarianceThreshold suggested to remove:\n")
print(set(new_num_X.columns)^set(credit_rank_1.drop(['default'],axis=1).columns))
# Checking  default precentages for the suggested fields to remove
print("Checking  default precentages for the suggested fields to remove:\n")
print(credit_rank_1.groupby('Total_of_Total_zero_flag')['default'].sum() / credit_rank_1.groupby('Total_of_Total_zero_flag')['default'].count())
print("\n-------------------------------\n")
print(credit_rank_1.groupby('EDUCATION_Other')['default'].sum() / credit_rank_1.groupby('EDUCATION_Other')['default'].count())
print("\n-------------------------------\n")
print(credit_rank_1.groupby('MARRIAGE_Other')['default'].sum() / credit_rank_1.groupby('MARRIAGE_Other')['default'].count())
print("\n-------------------------------\n")
print(credit_rank_1.groupby('Total_of_Total_one_flag')['default'].sum() / credit_rank_1.groupby('Total_of_Total_one_flag')['default'].count())
print("\n-------------------------------\n")
print(credit_rank_1.groupby('Total_Bills_LIMIT_BAL_flag')['default'].sum() / credit_rank_1.groupby('Total_Bills_LIMIT_BAL_flag')['default'].count())
# After the VarianceThreshold of 5% thete were 3 fields that it told us to delte.
# We decide to delete only the field of MARRIAGE_Other because only in that field we see a big differance in the default.

credit_rank.drop(['MARRIAGE_Other','Total_Bills_LIMIT_BAL_flag','Total_of_Total_one_flag','Total_of_Total_zero_flag'],axis=1,inplace=True)
credit_rank_1.drop(['MARRIAGE_Other','Total_Bills_LIMIT_BAL_flag','Total_of_Total_one_flag','Total_of_Total_zero_flag'],axis=1,inplace=True)
# List of chosen features and thier P_value grades
f_x=X[['trend_1','mean_trend','dev*limit_bal_']]
#f_x=X[X.columns[[0,1,2,3,4,5,11,15,16,17,18,19,20,23,24,25,26,27,28,29]]] # maybe pay1 and mean_trend trend1 are categogial with minus we have to scale to delete the minus
f_x.head()

fclass=f_classif(f_x,y)
p_values1=pd.Series(fclass[1],index=f_x.columns)
p_values1.sort_values(ascending=True,inplace=True)

print("List of quntity fields with sorted P_value grades:\n")
print(p_values1)
# Categorial field will be tested by Chi square test
# 12 13 14 are <0
chi_x=X[['Total_Delays','Three_Months_Delays','Total_Roll','Three_Months_Roll','count_out_of_bill','count_out_of_bill_3_months']]#,'trend_1','mean_trend']]
#chi_x=X[X.columns[[7,8,9,10,11,21,22,30,31,32,33,34,35,36,37,38,39,40,41,42]]]
chis2=chi2(chi_x,y)

p_values=pd.Series(chis2[1],index=chi_x.columns)
p_values.sort_values(ascending=True,inplace=True)

print("List of categorial fields with sorted P_value grades:\n")
p_values
X.head()
# Splitting data to X and y
X = credit_rank_1.drop('default', axis=1)
y = credit_rank_1.default
# Choosing the most relevant features
selected = X#.drop('Pay_from_Bill_1_round',axis=1)

#Using Feature Scaling
Scaler = StandardScaler().fit(selected)
selected_Scaled = pd.DataFrame(Scaler.transform(selected), columns=selected.columns)

# Splitting data to train and test
# X_train, X_test, y_train, y_test = split(selected_Scaled,y,train_size=0.7,random_state=12345,stratify=y)
X_train, X_test, y_train, y_test = split(selected_Scaled,y,train_size=0.7,random_state=12345,stratify=y)
# Logistic Regression model

log_reg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1, class_weight='balanced', random_state=12345,
                   max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None,solver='newton-cg')

# fitting X_train and y_train
logistic=log_reg.fit(X_train, y_train)
# Coefficient & intercept values
print("coef=",list(logistic.coef_))
print("\n******************************\n")
print(selected.columns)
print("\n*******************************\n")
print("intercept= ",logistic.intercept_)
# Using cross validation for testing the train data
Logistic_CV = StratifiedShuffleSplit(n_splits=7, train_size=0.7, test_size=0.3)

# We chose to use roc_auc score for testing our model
scores = cross_val_score(logistic, X_train, y_train, cv=Logistic_CV, scoring='roc_auc')

# The 7 cross validation scores
print("Scores : " + (7 * " {:.3f} ").format( *scores))

# mean score of the 7 cross validation
mean_scores = "%.3f" % stat.mean(scores)
print("Mean Scores: " ,mean_scores)
# Prediction using X_train
y_train_pred = logistic.predict(X_train)
y_test_pred = logistic.predict(X_test)
cm = confusion_matrix(y_true=y_test,
                      y_pred=y_test_pred)
pd.DataFrame(cm, 
             index=log_reg.classes_, 
             columns=log_reg.classes_)
# Using classification report for checking precision,recall,f1-score,support
print(classification_report(y_true=y_test,y_pred=y_test_pred))
# Using predict proba to check the probabilty of each row for default
y_test_pred_proba = pd.DataFrame(logistic.predict_proba(X_test), columns=logistic.classes_)

# Checking rows our model predicted as default 0 but actualy are deault 1
AA=y_test_pred_proba.join(X_test).join(y_test)
AA
# ROC scores by TPR,FPR,Treshold
y_test_desc = y_test.replace(1,'default').replace(0,'no-default')

scores = logistic.predict_proba(X_test)[:, 1]

fpr_log, tpr_log, thresholds_log = roc_curve(y_test_desc, scores, pos_label='default')
res_LR = pd.DataFrame({'FPR': fpr_log, 'TPR': tpr_log, 'Threshold': thresholds_log})
res_LR[['TPR', 'FPR', 'Threshold']][::200]
# ROC Plot Graph
plt.plot(fpr_log, tpr_log, '-o')
plt.title('ROC')
plt.xlabel('FPR (False Positive Rate = 1-specificity)')
plt.ylabel('TPR (True Positive Rate = sensitivity)')
plt.xlim([0, 1])
plt.ylim([0, 1])
# AUC Score
round(roc_auc_score(y_test_desc=='default', scores),4)
# Choosing the Hyper-Parameters values for grid search
LR_params_gs = {'solver':        ['lbfgs', 'newton-cg', 'sag', 'saga'],
                'C':             [0.01,0.05,0.1,1,10,100,500,1000],
                'multi_class':   ['auto','multinomial'],
                'class_weight':  ['None','balanced'],
                'fit_intercept': [True, False]}

# Fitting the grid search
logistic_reg_gs = GridSearchCV(logistic, LR_params_gs, cv=7,scoring='roc_auc')
logistic_reg_gs.fit(X_train, y_train)

# Best parameters
print("Best parameters:", logistic_reg_gs.best_params_)
print("\n*******************************************************************\n")
# Train & Test scores
print('Train Score: ',logistic_reg_gs.score(X_train, y_train))
print('Test Score: ',logistic_reg_gs.score(X_test, y_test))
# Splitting data to X and y
X = credit_rank_1.drop('default', axis=1)
y = credit_rank_1.default
#------------------------------------------
# Choosing the most relevant features
selected=X[['Three_Months_Delays','PAY_LM','dev*limit_bal_','LIMIT_BAL','Total_Delays','count_out_of_bill',	'mean_trend','max_delay','EDUCATION_Graduate School','Total_Roll','no_bill_flag',	'trend_1','AGE_round','Deviation_round']]

# Splitting data to train and test
X_train, X_test, y_train, y_test = split(selected,y,train_size=0.7,random_state=12345,stratify=y) # selected instead of X
#------------------------------------------
# Decision Tree model
DT = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=6, min_samples_split=2, min_samples_leaf=70, 
min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
min_impurity_split=None, class_weight='balanced', presort='deprecated', ccp_alpha=0.0)

# fitting X_train and y_train
Decision_Tree = DT.fit(X_train, y_train)
#-------------------------------------------
# Decision Tree visualization
def visualize_tree(model, md=5, width=1800):
    dot_data = StringIO()  
    export_graphviz(model, out_file=dot_data, feature_names=X_train.columns, max_depth=md)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]  
    return Image(graph.create_png(), width=width)

visualize_tree(Decision_Tree, md=3, width=1800)
#-------------------------------------------
def get_feature_importance(clsf, ftrs):
    imp = clsf.feature_importances_.tolist()
    feat = ftrs
    result = pd.DataFrame({'feat':feat,'score':imp})
    result = result.sort_values(by=['score'],ascending=False)
    return result

get_feature_importance(Decision_Tree, X_train.columns)
#---------------------------------------------
# Using cross validation for testing the train data
Decision_Tree_CV = StratifiedShuffleSplit(n_splits=7, train_size=0.7, test_size=0.3)

# We chose to use roc_auc score for testing our model
scores = cross_val_score(Decision_Tree, X_train, y_train, cv=Decision_Tree_CV, scoring='roc_auc')

# The 7 cross validation scores
print("Scores : " + (7 * " {:.3f} ").format( *scores))

# mean score of the 7 cross validation
mean_scores = "%.3f" % stat.mean(scores)
print("Mean Scores: " ,mean_scores)

cm = confusion_matrix(y_true=y_train,
                      y_pred=y_train_pred)
print(cm)

pd.DataFrame(cm, 
             index=DT.classes_, 
             columns=DT.classes_)

print(classification_report(y_true=y_train,y_pred=y_train_pred))
# Prediction using X_train
y_train_pred = Decision_Tree.predict(X_train)
y_test_pred = Decision_Tree.predict(X_test)
#-------------------------------------------
cm = confusion_matrix(y_true=y_test,
                      y_pred=y_test_pred)
pd.DataFrame(cm, 
             index=DT.classes_, 
             columns=DT.classes_)
#-------------------------------------------
# Using classification report for checking precision,recall,f1-score,support
print(classification_report(y_true=y_test,y_pred=y_test_pred))
#------------------------------------------
# Using predict proba to check the probabilty of each row for default
y_test_pred_proba = pd.DataFrame(Decision_Tree.predict_proba(X_test), columns=Decision_Tree.classes_)

# Checking rows our model predicted as default 0 but actualy are deault 1
AA=y_test_pred_proba.join(X_test).join(y_test)
AA.head()
# ROC scores by TPR,FPR,Treshold
y_test_desc = y_test.replace(1,'default').replace(0,'no-default')

scores = Decision_Tree.predict_proba(X_test)[:, 1]

fpr_DT, tpr_DT, thresholds_DT = roc_curve(y_test_desc, scores, pos_label='default')
res_DT = pd.DataFrame({'FPR': fpr_DT, 'TPR': tpr_DT, 'Threshold': thresholds_DT})
res_DT[['TPR', 'FPR', 'Threshold']][::5]

# ROC Plot Graph
plt.plot(fpr_DT, tpr_DT, '-o')
plt.title('ROC')
plt.xlabel('FPR (False Positive Rate = 1-specificity)')
plt.ylabel('TPR (True Positive Rate = sensitivity)')
plt.xlim([0, 1])
plt.ylim([0, 1])
# AUC Score
round(roc_auc_score(y_test_desc=='default', scores),3)
# Choosing the Hyper-Parameters values for grid search
DT_params_gs = {  'max_depth':           [3,4,5,6,7,8], 
                   'min_samples_split':   [2,5,10,20], 
                   'min_samples_leaf':    [25,50,70], 
                   'class_weight':        ['balanced']}

# Fitting the grid search
Decision_Tree_gs = GridSearchCV(Decision_Tree, DT_params_gs, cv=3,scoring='roc_auc')
Decision_Tree_gs.fit(X_train, y_train)

# Best parameters
print("Best parameters:", Decision_Tree_gs.best_params_)
print("\n*******************************************************************\n")
# Train & Test scores
print('Train Score: ',Decision_Tree_gs.score(X_train, y_train))
print('Test Score: ',Decision_Tree_gs.score(X_test, y_test))
# Splitting data to X and y
X = credit_rank.drop('default', axis=1)
y = credit_rank.default
# Choosing the most relevant features
selected=X[['Three_Months_Delays','PAY_LM','dev*limit_bal_','LIMIT_BAL','Total_Delays','count_out_of_bill',	'mean_trend','max_delay','Total_Roll','no_bill_flag','trend_1']]

# Using Feature Scaling 
Scaler = StandardScaler().fit(selected)
selected_Scaled = pd.DataFrame(Scaler.transform(selected), columns=selected.columns)

# Splitting data to train and testselected_Scaled
X_train, X_test, y_train, y_test = split(selected_Scaled,y,train_size=0.7,random_state=12345,stratify=y) # selected instead of X
# Knn model
KNN = KNeighborsClassifier(metric='minkowski', n_neighbors= 500, p= 1)

# fitting X_train and y_train
KNN_fit = KNN.fit(X_train, y_train)
# Using cross validation for testing the train data
Knn_CV = StratifiedShuffleSplit(n_splits=7, train_size=0.7, test_size=0.3)

# We chose to use roc_auc score for testing our model
scores = cross_val_score(KNN_fit, X_train, y_train, cv=Knn_CV, scoring='roc_auc')

# The 7 cross validation scores
print("Scores : " + (7 * " {:.3f} ").format( *scores))

# mean score of the 7 cross validation
mean_scores = "%.3f" % stat.mean(scores)
print("Mean Scores: " ,mean_scores)

cm = confusion_matrix(y_true=y_train,
                      y_pred=y_train_pred)
print(cm)

pd.DataFrame(cm, 
             index=KNN.classes_, 
             columns=KNN.classes_)

print(classification_report(y_true=y_train,y_pred=y_train_pred))
# Prediction using X_train
y_train_pred = KNN_fit.predict(X_train)
y_test_pred = KNN_fit.predict(X_test)
cm = confusion_matrix(y_true=y_test,
                      y_pred=y_test_pred)
pd.DataFrame(cm, 
             index=KNN_fit.classes_, 
             columns=KNN_fit.classes_)
# Using classification report for checking precision,recall,f1-score,support
print(classification_report(y_true=y_test,y_pred=y_test_pred))
# accuracy score
round(accuracy_score(y_true=y_test,y_pred=y_test_pred),3)
# Using predict proba to check the probabilty of each row for default
y_test_pred_proba = pd.DataFrame(KNN_fit.predict_proba(X_test), columns=KNN_fit.classes_)

# Checking rows our model predicted as default 0 but actualy are deault 1
AA=y_test_pred_proba.join(X_test).join(y_test)
AA.head()
# ROC scores by TPR,FPR,Treshold
y_test_desc = y_test.replace(1,'default').replace(0,'no-default')

scores = KNN_fit.predict_proba(X_test)[:, 1]

fpr_KNN, tpr_KNN, thresholds_KNN = roc_curve(y_test_desc, scores, pos_label='default')
res_KNN = pd.DataFrame({'FPR': fpr_KNN, 'TPR': tpr_KNN, 'Threshold': thresholds_KNN})
res_KNN[['TPR', 'FPR', 'Threshold']]

# ROC Plot Graph
plt.plot(fpr_KNN, tpr_KNN, '-o')
plt.title('ROC')
plt.xlabel('FPR (False Positive Rate = 1-specificity)')
plt.ylabel('TPR (True Positive Rate = sensitivity)')
plt.xlim([0, 1])
plt.ylim([0, 1])
# AUC Score
round(roc_auc_score(y_test_desc=='default', scores),3)
# Choosing the Hyper-Parameters values for grid search
KNN_params_gs = {'n_neighbors':  [1,5,10,25,50,100,500,1000],
                 'metric':       ['minkowski', 'hamming', 'cosine'],
                 'p':            [1,2,3]}         # p = 2 is euclidean (p = 1 is manhattan)

# Fitting the grid search
KNN_params_gs = GridSearchCV(KNN_fit, KNN_params_gs, cv=2,scoring='roc_auc')
KNN_params_gs.fit(X_train, y_train)

# Best parameters
print("Best parameters:", KNN_params_gs.best_params_)
print("\n*******************************************************************\n")
# Train & Test scores
print('Train Score: ',KNN_params_gs.score(X_train, y_train))
print('Test Score: ',KNN_params_gs.score(X_test, y_test))

# ROC Plot Graph
plt.plot(fpr_log, tpr_log, '-o')
plt.plot(fpr_DT, tpr_DT, '-o')
plt.plot(fpr_KNN, tpr_KNN, '-o')
plt.title('ROC')
plt.xlabel('FPR (False Positive Rate = 1-specificity)')
plt.ylabel('TPR (True Positive Rate = sensitivity)')
plt.xlim([0, 1])
plt.ylim([0, 1])
# The 3 models with their best Hyper Parameters
clf1 = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1, class_weight='balanced', random_state=12345,
       max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None,solver='newton-cg')

clf2 = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=6, min_samples_split=2, min_samples_leaf=70, 
       min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
       min_impurity_split=None, class_weight='balanced', presort='deprecated', ccp_alpha=0.0)

clf3 = KNeighborsClassifier(metric='minkowski', n_neighbors= 500, p= 1)

classifiers = [('LR', clf1), ('DT', clf2), ('KNN', clf3)]
# fitting the voting model (voting=hard)
clf_voting = VotingClassifier(estimators=classifiers,
                              voting='hard')
clf_voting.fit(X_train, y_train)

# Scores:
y_test_desc = y_test.replace(1,'default').replace(0,'no-default')
scores = clf_voting.predict(X_test)

print()
print("--------------------------------------------------------------------")
print()
print("Test AUC Score: ",round(roc_auc_score(y_test_desc=='default', scores),3))
print()
print("--------------------------------------------------------------------")
print()
print("Confusion Matrix:")
cm = confusion_matrix(y_true=y_test,y_pred=scores)
print(pd.DataFrame(cm, index=DT.classes_, columns=DT.classes_))
print()
print("--------------------------------------------------------------------")
print()
print("Classification Report:")
print(classification_report(y_true=y_test,y_pred=y_test_pred))
# fitting the voting model (voting=soft)
clf_voting = VotingClassifier(estimators=classifiers,
                              voting='soft')
clf_voting.fit(X_train, y_train)

# Scores:
y_test_desc = y_test.replace(1,'default').replace(0,'no-default')
scores = clf_voting.predict(X_test)

print()
print("--------------------------------------------------------------------")
print()
print("Test AUC Score: ",round(roc_auc_score(y_test_desc=='default', scores),3))
print()
print("--------------------------------------------------------------------")
print()
print("Confusion Matrix:")
cm = confusion_matrix(y_true=y_test,y_pred=scores)
print(pd.DataFrame(cm, index=DT.classes_, columns=DT.classes_))
print()
print("--------------------------------------------------------------------")
print()
print("Classification Report:")
print(classification_report(y_true=y_test,y_pred=y_test_pred))

#bagging scores func
def bagging_scores(clf):
  clf_bagging = BaggingClassifier(base_estimator=clf, n_estimators=100)
  clf_bagging.fit(X_train, y_train)
  
  #Scores:
  y_test_desc = y_test.replace(1,'default').replace(0,'no-default')
  scores = clf_bagging.predict(X_test)
  print()
  print("--------------------------------------------------------------------")
  print()
  print("Test AUC Score: ",round(roc_auc_score(y_test_desc=='default', scores),3))
  print()
  print("--------------------------------------------------------------------")
  print()
  print("Confusion Matrix:")
  cm = confusion_matrix(y_true=y_test,y_pred=scores)
  print(pd.DataFrame(cm, index=DT.classes_, columns=DT.classes_))
  print()
  print("--------------------------------------------------------------------")
  print()
  print("Classification Report:")
  print(classification_report(y_true=y_test,y_pred=y_test_pred))
# Using the function for the 3 Models:
print("======================================================================")
print("Logistic Regression Bagging Score:")
print("======================================================================")
bagging_scores(clf1)
print("======================================================================")
print("Decision Tree Bagging Score:")
print("======================================================================")
bagging_scores(clf2)
print("======================================================================")
print("KNN Bagging Score:")
print("======================================================================")
#bagging_scores(clf3)
def adaboost_scores(clf):
  clf_adaboost = AdaBoostClassifier(base_estimator=clf,n_estimators=200,learning_rate=0.01)
  clf_adaboost.fit(X_train, y_train)

  #Scores:
  y_test_desc = y_test.replace(1,'default').replace(0,'no-default')
  scores = clf_adaboost.predict(X_test)
  print()
  print("--------------------------------------------------------------------")
  print()
  print("Test AUC Score: ",round(roc_auc_score(y_test_desc=='default', scores),3))
  print()
  print("--------------------------------------------------------------------")
  print()
  print("Confusion Matrix:")
  cm = confusion_matrix(y_true=y_test,y_pred=scores)
  print(pd.DataFrame(cm, index=DT.classes_, columns=DT.classes_))
  print()
  print("--------------------------------------------------------------------")
  print()
  print("Classification Report:")
  print(classification_report(y_true=y_test,y_pred=y_test_pred))
# Using the function for the 3 Models:
print("======================================================================")
print("Logistic Regression Bagging Score:")
print("======================================================================")
adaboost_scores(clf1)
print("======================================================================")
print("Decision Tree Bagging Score:")
print("======================================================================")
adaboost_scores(clf2)
print("======================================================================")
print("KNN Bagging Score:")
print("======================================================================")
#adaboost_scores(clf3)
clf_GB = GradientBoostingClassifier(max_depth=3, n_estimators=200, learning_rate=0.01)
clf_GB.fit(X_train, y_train)

#Scores:
y_test_desc = y_test.replace(1,'default').replace(0,'no-default')
scores = clf_GB.predict(X_test)
print()
print("--------------------------------------------------------------------")
print()
print("Test AUC Score: ",round(roc_auc_score(y_test_desc=='default', scores),3))
print()
print("--------------------------------------------------------------------")
print()
print("Confusion Matrix:")
cm = confusion_matrix(y_true=y_test,y_pred=scores)
print(pd.DataFrame(cm, index=DT.classes_, columns=DT.classes_))
print()
print("--------------------------------------------------------------------")
print()
print("Classification Report:")
print(classification_report(y_true=y_test,y_pred=y_test_pred))