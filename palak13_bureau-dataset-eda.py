import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
bureau_df = pd.read_csv("../input/bureau.csv")
bureau_df.head(15)
bureau_df = bureau_df.iloc[0:500000,0:]
print(bureau_df.shape)
print('Number of NAs in the dataset is:\n',bureau_df.isnull().sum())
df1 = bureau_df[["CREDIT_ACTIVE","DAYS_CREDIT_ENDDATE","DAYS_ENDDATE_FACT","CNT_CREDIT_PROLONG",
                          "AMT_CREDIT_SUM_LIMIT","AMT_CREDIT_SUM_OVERDUE","CREDIT_TYPE","DAYS_CREDIT_UPDATE"]]
df1.head(10)
print('Number of NAs in the dataset is: \n',df1.isnull().sum())
df1 = df1.dropna(thresh=7)
print('Number of NAs in the dataset is:\n',df1.isnull().sum())
df1.shape
sns.jointplot(x="AMT_CREDIT_SUM_LIMIT", y="AMT_CREDIT_SUM_OVERDUE", data=df1)
plt.show()
sns.stripplot(x="CNT_CREDIT_PROLONG", y="CREDIT_TYPE", data=df1)
plt.show()
df2 = df1[df1.DAYS_CREDIT_ENDDATE > 0]
sns.countplot(x="DAYS_CREDIT_ENDDATE", data=df2, palette="Blues")
plt.show()
df2= df2.groupby(['DAYS_CREDIT_ENDDATE'])
df2.size()