# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#what the data looks like
data  = pd.read_csv("../input/restaurant-scores-lives-standard.csv")
data.head(10)
data['inspection_type'].value_counts()


plt.hist(data['inspection_type'].value_counts())
plt.show()

print(data['risk_category'].value_counts())
data["business_postal_code"].value_counts()
import seaborn as sns
%matplotlib inline 
df = data[["business_postal_code",'risk_category']]
sns.countplot(x='risk_category',data=df)


data['inspection_type'].value_counts()
sns.countplot(x="inspection_type", hue = "risk_category", data=data)
simple_inspection_type = {'inspection_type':{'Routine - Unscheduled':'Unexpected',
                                             'Reinspection/Followup':'Expected',
                                             'Complaint':'Expected',
                                             'New Ownership':'Expected',
                                             'New Construction':'Expected',
                                             'Non-inspection site visit':'Expected',
                                             'Structural Inspection':'Expected',
                                             'New Ownership - Followup':'Expected',
                                             'Complaint Reinspection/Followup':'Expected',
                                             'Foodborne Illness Investigation':'Expected',
                                             'Routine - Scheduled':'Expected',
                                             'Special Event':'Expected',
                                             'Administrative or Document Review':'Expected',
                                             'Home Environmental Assessment':'Expected',
                                             'Community Health Assessment':'Expected'
                                             }}

data_copy = data.copy()
data_copy.replace(simple_inspection_type,inplace = True)
sns.countplot(x="inspection_type", hue = "risk_category", data=data_copy)
data["business_postal_code"].unique()
len(data["business_postal_code"].unique())
plt.figure(figsize=(15,10))
ax = sns.countplot(x="business_postal_code", hue = "risk_category", data=data_copy)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()
