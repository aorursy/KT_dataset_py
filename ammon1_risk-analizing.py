import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
from matplotlib import pyplot
from matplotlib import rcParams
import seaborn as sns
import matplotlib.pyplot as pl
df=pd.read_csv("../input/restaurant-and-market-health-inspections.csv")
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()*100/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
df.columns.values
print('serial_number ',df.serial_number.unique().shape)
print('activity_date ',df.activity_date.unique().shape)
print('facility_name ',df.facility_name.unique().shape)#to remove
print('score ',df.score.unique().shape)
print('grade ',df.grade.unique().shape)
print('service_code ',df.service_code.unique().shape)
print('service_description ',df.service_description.unique().shape)
print('employee_id ',df.employee_id.unique().shape)
print('facility_address ',df.facility_address.unique().shape)#to remove
print('facility_id',df.facility_id.unique().shape)#to remove
print('facility_state',df.facility_state.unique().shape)#to remove
print('facility_zip',df.facility_zip.unique().shape)
print('owner_id ',df.owner_id.unique().shape)#to remove
print('owner_name',df.owner_name.unique().shape)#to remove
print('pe_description',df.pe_description.unique().shape)#VIP
print('program_element_pe',df.program_element_pe.unique().shape)#VIP
print('program_name',df.program_name.unique().shape)
print('program_status',df.program_status.unique().shape)
print('record_id',df.record_id.unique().shape)
df=df.sort_values(by=['facility_name'])
fe_v=df.loc[:,'facility_name'].values
print(fe_v[0:20])
fd=df.sort_values(by=['owner_id'])
fd.head(20)
df_nd=df.drop_duplicates('owner_id',keep='first')
df_nd=df_nd.drop_duplicates('owner_name',keep='first')
df_nd.head()
print('serial_number ',df_nd.serial_number.unique().shape)
print('activity_date ',df_nd.activity_date.unique().shape)
print('facility_name ',df_nd.facility_name.unique().shape)#to remove
print('score ',df_nd.score.unique().shape)
print('grade ',df_nd.grade.unique().shape)
print('service_code ',df_nd.service_code.unique().shape)
print('service_description ',df_nd.service_description.unique().shape)
print('employee_id ',df_nd.employee_id.unique().shape)
print('facility_address ',df_nd.facility_address.unique().shape)#to remove
print('facility_id',df_nd.facility_id.unique().shape)#to remove
print('facility_state',df_nd.facility_state.unique().shape)#to remove
print('facility_zip',df_nd.facility_zip.unique().shape)
print('owner_id ',df_nd.owner_id.unique().shape)#to remove
print('owner_name',df_nd.owner_name.unique().shape)#to remove
print('pe_description',df_nd.pe_description.unique().shape)#VIP
print('program_element_pe',df_nd.program_element_pe.unique().shape)#VIP
print('program_name',df_nd.program_name.unique().shape)
print('program_status',df_nd.program_status.unique().shape)
print('record_id',df_nd.record_id.unique().shape)
df2=df_nd[['score','grade','service_code','service_description','employee_id',
        'pe_description','program_element_pe','program_status','facility_zip','owner_id','facility_address']].copy()
df2['facility_zip'] = df2['facility_zip'].str.extract('(\d+)', expand=False)
df2['facility_zip']=pd.to_numeric(df2['facility_zip'])  
df2['owner_id'].astype(str) 
df2['owner_id'] = df2['owner_id'].str.extract('(\d+)', expand=False)
df2['owner_id']=pd.to_numeric(df2['owner_id'])  
df2['employee_id'].astype(str) 
df2['employee_id'] = df2['employee_id'].str.extract('(\d+)', expand=False)
df2['employee_id']=pd.to_numeric(df2['employee_id'])     
df2.head()
df_sc=df[df.service_code>1]
df_sc.loc[:,'service_code'].values.shape
df2.info()
df2 = pd.get_dummies(df2, prefix='grade_', columns=['grade'])
df2 = pd.get_dummies(df2, prefix='pe_description_', columns=['pe_description'])
df2 = pd.get_dummies(df2, prefix='program_status_', columns=['program_status'])


f, ax = pl.subplots(figsize=(10, 8))
corr = df2.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
df_1=df.sort_values(by=['owner_id'])
df_1['year']=df_1.activity_date.str[0:4]
df_1['year']=pd.to_numeric(df_1['year'])  
df_1['month']=df_1.activity_date.str[5:7]
df_1['month']=pd.to_numeric(df_1['month'])  
df_1['date']=df_1['year']+df_1['month']/12
df_1.head()
ax = sns.boxplot(x="year", y="score",data=df_1, palette="Set3")
ax1 = sns.boxplot(x="date", y="score",data=df_1, palette="Set3")