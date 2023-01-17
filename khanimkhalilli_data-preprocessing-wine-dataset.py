import pandas as pd
import io
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
!pip install lightgbm
from scipy import stats
import scipy.stats as ss
from lightgbm import LGBMRegressor,LGBMClassifier
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv")
data.head()
data.info()
data.hist();
sns.scatterplot(x="price",y="region_2",data=data);
sns.kdeplot(data.price,shade=True);
sns.barplot(x="points",y="price",data=data);
sparse_columns=[]
for col in ["points","price"]:
  if (data[col].mode()[0]==data[col].quantile(0.01)==data[col].quantile(0.25) or (data[col].mode()[0]==data[col].quantile(0.99)==data[col].quantile(0.75))):
    sparse_columns.append(col)
sparse_columns
def boxplot_for_outlier(df,columns):
  for col in columns:
    fig, ax =plt.subplots(figsize=(7,5))
    sns.boxplot(x = df[col], palette="Set2")
boxplot_for_outlier(data,["points","price"])
def outliers_detection(df,columns):
  for col in columns:
    q1=df[col].describe()[4]
    q3=df[col].describe()[6]
    iqr=q3-q1
    lowerbound = q1 - (1.5 * iqr)
    upperbound = q3 + (1.5 * iqr)
    number_of_outlier = df.loc[(df.loc[:,col]<lowerbound)| (df.loc[:,col]>upperbound)].shape[0]
    if(number_of_outlier>0):
      print(number_of_outlier," outlier values cleared in" ,col)
      df.loc[(df.loc[:,col]<lowerbound),col] =  lowerbound*0.75
      df.loc[(df.loc[:,col]>upperbound),col] =  upperbound*1.25
data_outliers=data.copy()
outliers_detection(data_outliers,["points","price"])
boxplot_for_outlier(data_outliers,["points","price"])
def Missing_Values(data):
    variable_name=[]
    total_value=[]
    total_missing_value=[]
    missing_value_rate=[]
    unique_value_list=[]
    total_unique_value=[]
    data_type=[]
    for col in data.columns:
        variable_name.append(col)
        data_type.append(data[col].dtype)
        total_value.append(data[col].shape[0])
        total_missing_value.append(data[col].isnull().sum())
        missing_value_rate.append(round(data[col].isnull().sum()/data[col].shape[0],5))
        unique_value_list.append(data[col].unique())
        total_unique_value.append(len(data[col].unique()))
    missing_data=pd.DataFrame({"Variable":variable_name,"Total_Value":total_value,\
                             "Total_Missing_Value":total_missing_value,"Missing_Value_Rate":missing_value_rate,
                             "Data_Type":data_type,"Unique_Value":unique_value_list,\
                               "Total_Unique_Value":total_unique_value})
    
    return missing_data.sort_values("Missing_Value_Rate",ascending=False)
data_info=Missing_Values(data_outliers)
data_info = data_info.set_index("Variable")
data_info
categorical_columns=data_info[data_info["Data_Type"]=="object"].index
numerical_columns=data_info[(data_info["Data_Type"]=="float64") | (data_info["Data_Type"]=="int64")].index
len(categorical_columns),len(numerical_columns)
import missingno as msno
msno.bar(data);
msno.matrix(data);
msno.heatmap(data);
def simple_imputer(df,columns):
  for col in columns:
    total_nan=int(df[col].isnull().sum())
    if(col in categorical_columns):
  
      most_frequent_value=str(stats.mode(df[col])[0][0])

      df[col]=df[col].fillna(most_frequent_value)
      print("Total imputed in {} : {} ".format(col,total_nan))
    else:
      
      mean=df[col].mean()
      std=df[col].std()
      random_normal=np.random.normal(loc=mean,scale=std,size=total_nan) 
      df[col][df[col].isnull()]=random_normal
      print("Total imputed in {} : {} ".format(col,total_nan))
  return df
data_simple_imp=data_outliers.copy()
columns=list(data_info[(data_info["Missing_Value_Rate"]>0 )& (data_info["Missing_Value_Rate"]<0.1 ) ].index)
columns.append("designation")
data_simpe_imp=simple_imputer(data_simple_imp,columns)
Missing_Values(data_simple_imp)
from sklearn_pandas import CategoricalImputer
lst=list(data_info[(data_info["Missing_Value_Rate"]>0.1 )].index)
lst.remove("designation")
data_c_imputer=data_simple_imp.copy()
for col in lst:

  imputer = CategoricalImputer()
  data_c_imputer[col]=imputer.fit_transform(data_c_imputer[col])
Missing_Values(data_c_imputer)
"""lgbm_imputer = LGBMClassifier()
lst=list(data_info[(data_info["Missing_Value_Rate"]>0.1 )].index)
lst.remove("designation")
df_mbi = data_simpe_imp.copy()

import re
df_mbi = df_mbi.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

for c in df_mbi.columns:
    col_type= df_mbi[c].dtype
    if col_type == "object" or col_type.name=="category":
        df_mbi[c] = df_mbi[c].astype("category")

for col in lst:
    y_col_nan_train = df_mbi.loc[df_mbi.loc[:,col].isnull(),col]
    y_col_train = df_mbi.loc[df_mbi.loc[:,col].notnull(),col]  
    X_col_nan_train = df_mbi.drop(columns=col,axis=1).loc[y_col_nan_train.index,:]
    X_col_train = df_mbi.drop(columns=col,axis=1).loc[y_col_train.index,:]
    print(col)
    lgbm_imputer.fit(X_col_train,y_col_train)
    y_col_pred_train = lgbm_imputer.predict(X_col_nan_train)

    df_mbi.loc[y_col_nan_train.index,col] = y_col_pred_train


    print(col," null sayisi:",df_mbi[col].isnull().sum())"""

#from scipy import stats
#import scipy.stats as ss
"""def cramers_v(columns,df):
    chi2_df=pd.DataFrame(columns=columns)
    for col in columns:
        lst=[]
        print(col)
        for col2 in columns:

            confusion_matrix = pd.crosstab(df[col],df[col2])
            chi2 = ss.chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2/n
            r,k = confusion_matrix.shape
            phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
            rcorr = r-((r-1)**2)/(n-1)
            kcorr = k-((k-1)**2)/(n-1)
            lst.append(np.sqrt(phi2corr/min((kcorr-1),(rcorr-1))))

        chi2_df.loc[col]=lst
    return chi2_df
categorical_columns=list(data.select_dtypes(include=['object']).columns)
categorical_columns.remove("description")
categorical_columns.remove("title")
chi2_rate =cramers_v(categorical_columns.copy(),data.copy())"""
#Missing_Values(data)
#f, ax = plt.subplots(figsize=(20, 30))
#sns.heatmap(chi2_rate, annot=True,fmt=".1f")
#plt.show()