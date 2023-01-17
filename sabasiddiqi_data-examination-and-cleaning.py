import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plots
import seaborn as sns
%matplotlib inline
import os
import warnings
warnings.filterwarnings('ignore')
print(os.listdir("../input"))
data = pd.read_csv('../input/horse.csv')
data.head()
print("Shape of data (samples, features): ",data.shape)
data.dtypes.value_counts()
nan_per=data.isna().sum()/len(data)*100
plt.bar(range(len(nan_per)),nan_per)
plt.xlabel('Features')
plt.ylabel('% of NAN values')
plt.plot([0, 25], [40,40], 'r--', lw=1)
plt.xticks(list(range(len(data.columns))),list(data.columns.values),rotation='vertical')
obj_columns=[]
nonobj_columns=[]
for col in data.columns.values:
    if data[col].dtype=='object':
        obj_columns.append(col)
    else:
        nonobj_columns.append(col)
print(len(obj_columns)," Object Columns are \n",obj_columns,'\n')
print(len(nonobj_columns),"Non-object columns are \n",nonobj_columns)

data_obj=data[obj_columns]
data_nonobj=data[nonobj_columns]
print("Data Size Before Numerical NAN Column(>40%) Removal :",data_nonobj.shape)
for col in data_nonobj.columns.values:
    if (pd.isna(data_nonobj[col]).sum())>0:
        if pd.isna(data_nonobj[col]).sum() > (40/100*len(data_nonobj)):
            print(col,"removed")
            data_nonobj=data_nonobj.drop([col], axis=1)
        else:
            data_nonobj[col]=data_nonobj[col].fillna(data_nonobj[col].median())
print("Data Size After Numerical NAN Column(>40%) Removal :",data_nonobj.shape)
print("Data Size Before Categorical NAN Column(>40%) Removal :",data_obj.shape)
for col in data_obj.columns.values:
    if (pd.isna(data_obj[col]).sum())>0:
        if pd.isna(data_obj[col]).sum() > (40/100*len(data_nonobj)):
            print(col,"removed")
            data_obj=data_obj.drop([col], axis=1)
        else:
            data_obj[col]=data_obj[col].fillna(data_obj[col].mode()[0])
print("Data Size After Categorical NAN Column(>40%) Removal :",data_obj.shape)
for col in data_obj.columns.values:
    data_obj[col]=data_obj[col].astype('category').cat.codes
data_merge=pd.concat([data_nonobj,data_obj],axis=1)

target=data['outcome']
print(target.value_counts())
target=data_merge['outcome']
print(target.value_counts())
train_corr=data_merge.corr()
sns.heatmap(train_corr, vmax=0.8)
corr_values=train_corr['outcome'].sort_values(ascending=False)
corr_values=abs(corr_values).sort_values(ascending=False)
print("Correlation of mentioned features wrt outcome in ascending order")
print(abs(corr_values).sort_values(ascending=False))
print("Data Size Before Correlated Column Removal :",data_merge.shape)

for col in range(len(corr_values)):
        if abs(corr_values[col]) < 0.1:
            data_merge=data_merge.drop([corr_values.index[col]], axis=1)
            print(corr_values.index[col],"removed")
print("Data Size After Correlated Column Removal :",data_merge.shape)
#packed_cell_volume 
col='packed_cell_volume'
fig,(ax1,ax2)=plt.subplots(1,2, figsize=[20,10])

y=data_merge[col][target==2]
x=data_merge['outcome'][target==2]
y.plot.hist(ax=ax2)
sns.kdeplot(y,ax=ax1)

y=data_merge[col][target==0]
x=data_merge['outcome'][target==0]
y.plot.hist(ax=ax2)
sns.kdeplot(y,ax=ax1)

y=data_merge[col][target==1]
x=data_merge['outcome'][target==1]
y.plot.hist(ax=ax2)
sns.kdeplot(y,ax=ax1)

plt.title(col)
ax1.legend(['lived','died','euthanized'])
ax2.legend(['lived','died','euthanized'])
plt.show()
#pulse 
col='pulse'
fig,(ax1,ax2)=plt.subplots(1,2, figsize=[20,10])
y=data_merge[col][target==2]
x=data_merge['outcome'][target==2]
y.plot.hist(ax=ax2)
sns.kdeplot(y,ax=ax1)
y=data_merge[col][target==0]
x=data_merge['outcome'][target==0]
y.plot.hist(ax=ax2)
sns.kdeplot(y,ax=ax1)
y=data_merge[col][target==1]
x=data_merge['outcome'][target==1]
y.plot.hist(ax=ax2)
sns.kdeplot(y,ax=ax1)
plt.title(col)
ax1.legend(['lived','died','euthanized'])
ax2.legend(['lived','died','euthanized'])
plt.show()