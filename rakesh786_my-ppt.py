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
benifits=pd.read_csv("../input/BenefitsCostSharing.csv")
(benifits.isnull().sum()/len(benifits))*100
req_benifits_col=benifits[['PlanId','BenefitName','BusinessYear','Exclusions','Explanation','ImportDate','IsCovered','IssuerId','LimitQty','LimitUnit','MinimumStay','SourceName','StateCode']]
benifits[benifits['Exclusions'].notnull()]
(req_benifits_col.isnull().sum()/len(req_benifits_col))*100
req_benifits_col=req_benifits_col.drop(['Exclusions','LimitQty','MinimumStay','LimitUnit','Explanation'],axis=1)
(req_benifits_col.isnull().sum()/len(req_benifits_col))*100
req_benifits_col.shape
req_benifits_col=req_benifits_col.dropna()
(req_benifits_col.isnull().sum()/len(req_benifits_col))*100
Rate=pd.read_csv("../input/Rate.csv")
Rate.head()
(Rate.isnull().sum()/len(Rate))*100
Rate=Rate[['PlanId','BusinessYear','StateCode','IssuerId','SourceName','RatingAreaId','RateEffectiveDate','RateExpirationDate','Tobacco','Age','IndividualRate']]
req_benifits_col.head()
req_benifits_col.StateCode.value_counts()
req_benifits_col2016=req_benifits_col[(req_benifits_col.BusinessYear==2016) & (req_benifits_col.BenefitName=='X-rays and Diagnostic Imaging')]
req_benifits_col2016.shape
req_benifits_col2016
Rate_2016=Rate[Rate.BusinessYear==2016]
Rate_2016.shape
req_benifits_col.info()
Rate.info()
req_benifits_col2016.merge(Rate_2016,how='left',on='PlanId')