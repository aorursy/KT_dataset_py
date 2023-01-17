# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
Inspection=pd.read_csv('../input/restaurant-and-market-health-inspections.csv')
Violation=pd.read_csv('../input/restaurant-and-market-health-violations.csv')


Inspection.head(10)
#Violation.head()
#Inspection.isnull().sum()
Inspection.program_element_pe.unique()
Inspection.program_name.astype('category').value_counts()
Inspection.groupby('program_element_pe').mean()[['score']].sort_values(by=['score'],ascending=False).sample(n=50,replace=True).reset_index().plot.scatter(x='program_element_pe',y='score')
#col=Inspection.groupby('employee_id').mean()[['score']].sort_values(by=['score'],ascending=False)
#col=col.head(10)
#col=col.reset_index()
#col.plot.hist(x="employee_id",y="score")
Inspection['employee_id']=Inspection.employee_id.astype('category').cat.codes
#Inspection.head(20)
col=(Inspection
     .groupby(['employee_id','grade'])
     .mean()[['score']]
     .sort_values(by=['score'],ascending=False)
     .sample(n=20)
     .reset_index()
    )

#col.plot.hist(x="employee_id",y="score")
sns.scatterplot(y="score",x="employee_id",hue='grade' ,data=col)
Inspection['year']=Inspection.activity_date.str[0:4]
score_by_fac=pd.pivot_table(Inspection,index=['facility_name','year'],values='score')


(score_by_fac
.reset_index()
.groupby(['facility_name','year'])
.mean()[['score']]
.sample(n=20)
.plot.bar(stacked=True)
)
#Violation['violation_status'].unique()
#Violation['grade'].unique()
Violation.dtypes
#Violation
Inspection=pd.read_csv('../input/restaurant-and-market-health-inspections.csv')
Violation=pd.read_csv('../input/restaurant-and-market-health-violations.csv')
Newdata=Inspection.set_index('program_name').merge(Violation.set_index('program_name'))
Newdata.head()

Newdata.violation_status[Violation.violation_status == 'OUT OF COMPLIANCE'] = 1
Newdata.violation_status[Violation.violation_status == 'VIOLATION'] = 2
Newdata.violation_status[Violation.violation_status == 'HOUSING NON-CRITICAL'] = 3
Newdata.violation_status.astype(int)    
sns.pairplot(Newdata,vars=['points','violation_status'],hue = 'grade' )
col=Newdata.groupby(['employee_id','grade']).mean()[['score']].sort_values(by=['score'],ascending=False)
col=col.reset_index()
col['employee_id']=col.employee_id.astype('category').cat.codes
col=col.sample(n=20)
sns.swarmplot(x="employee_id", y="score", hue="grade", data=col);
Newdata.violation_description.astype('category').head(10).unique()
Violation_desc=Newdata.violation_description.astype('category').head(20).cat.codes.reset_index()
Violation_desc.dtypes
Violation_desc=Violation_desc.rename(columns={0:'cat_codes'})
Violation_desc
sns.countplot(x='cat_codes',data=Violation_desc);
#Violation_desc.set_index('cat_codes', inplace=True)
#Violation_desc.rename_axis("Violation description")
#Violation_desc.set_index(['Cat_Code','Count'],append=true)
#Violation_desc.drop("Violation description")
#.rename(index={"index:Cat Codes"})
#Newdata.dtypes
#Newdata["employee_id"]=Newdata.employee_id.str.extract('(\d+)', expand=False).astype(int)
#Newdata["owner_id"]=Newdata.owner_id.str.extract('(\d+)', expand=False).astype(int)
Newdata.head(20)
correlationdf=pd.get_dummies(Newdata,prefix="grade_",columns=["grade"])
correlationdf=pd.get_dummies(Newdata,prefix="pe_description_",columns=["pe_description"])
correlationdf=pd.get_dummies(Newdata,prefix="service_code_",columns=["service_code"])
correlationdf.head()
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(10, 8))
corr = correlationdf.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10 , as_cmap=True),
            square=True, ax=ax);
Newdata['year']=Newdata.activity_date.str[0:4]
Newdata['month']=Newdata.activity_date.str[5:7]
Newdata.month.unique()
    
plt.subplots(figsize=(10, 8))
sns.boxplot(x='year',y='score',data=Newdata);
#sns.violinplot(x='year',y='score',data=Newdata)
#sns.boxplot(x='year',y='score',data=Newdata
plt.subplots(figsize=(10, 8))
sns.boxplot(x='month',y='score',hue='year',data=Newdata);

