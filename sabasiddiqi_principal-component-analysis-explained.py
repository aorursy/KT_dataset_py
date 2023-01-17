import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plots
#import seaborn as sns #fo
%matplotlib inline
import os
import warnings
warnings.filterwarnings('ignore')
print(os.listdir("../input"))
data = pd.read_csv('../input/horse.csv')
data.head()
#print("Shape of data (samples, features): ",data.shape)
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
#print(len(obj_columns)," Object Columns are \n",obj_columns,'\n')
#print(len(nonobj_columns),"Non-object columns are \n",nonobj_columns)

data_obj=data[obj_columns]
data_nonobj=data[nonobj_columns]

#print("Data Size Before Numerical NAN Column(>40%) Removal :",data_nonobj.shape)
for col in data_nonobj.columns.values:
    if (pd.isna(data_nonobj[col]).sum())>0:
        if pd.isna(data_nonobj[col]).sum() > (40/100*len(data_nonobj)):
            #print(col,"removed")
            data_nonobj=data_nonobj.drop([col], axis=1)
        else:
            data_nonobj[col]=data_nonobj[col].fillna(data_nonobj[col].median())
#print("Data Size After Numerical NAN Column(>40%) Removal :",data_nonobj.shape)

for col in data_obj.columns.values:
    data_obj[col]=data_obj[col].astype('category').cat.codes
data_merge=pd.concat([data_nonobj,data_obj],axis=1)

target=data['outcome']
temp=target
#print(target.value_counts())
target=data_merge['outcome']
#print(target.value_counts())

train_corr=data_merge.corr()
#sns.heatmap(train_corr, vmax=0.8)
corr_values=train_corr['outcome'].sort_values(ascending=False)
corr_values=abs(corr_values).sort_values(ascending=False)
#print("Correlation of mentioned features wrt outcome in ascending order")
#print(abs(corr_values).sort_values(ascending=False))

#print("Data Size Before Correlated Column Removal :",data_merge.shape)

for col in range(len(corr_values)):
        if abs(corr_values[col]) < 0.1:
            data_merge=data_merge.drop([corr_values.index[col]], axis=1)
            #print(corr_values.index[col],"removed")
#print("Data Size After Correlated Column Removal :",data_merge.shape)
data_merge.head()
data.describe()
from sklearn.preprocessing import StandardScaler
Xstd = StandardScaler().fit_transform(data_merge)
print('Covariance matrix: \n', np.cov(Xstd.T))
cov_mat = np.cov(Xstd.T)
eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
print('Eigen vectors \n',eigen_vectors)
print('\nEigen values \n',eigen_values)
pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
pairs.sort()
pairs.reverse()

print('Eigen Values in descending order:')
for i in pairs:
    print(i[0])
tot = sum(eigen_values)
var_per = [(i / tot)*100 for i in sorted(eigen_values, reverse=True)]
cum_var_per = np.cumsum(var_per)

plt.figure(figsize=(10,10))
x=['PC %s' %i for i in range(1,len(var_per))]
ind = np.arange(len(var_per)) 
plt.bar(ind,var_per)
plt.xticks(ind,x);
plt.plot(ind,cum_var_per,marker="o",color='orange')
plt.xticks(ind,x);

N=16
value=10
a = np.ndarray(shape = (N, 0))
for x in range(1,value):
    b=pairs[x][1].reshape(16,1)
    a = np.hstack((a,b))
print("Projection Matrix:\n",a)
Y = Xstd.dot(a)
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
for name in ('died', 'euthanized', 'lived'):
    plt.scatter(
        x=Y[temp==name,3],
        y=Y[temp==name,4],
    )
plt.legend( ('died', 'euthanized', 'lived'))
plt.title('After PCA')

plt.subplot(1,2,2)
for name in ('died', 'euthanized', 'lived'):
    plt.scatter(
        x=Xstd[temp==name,3],
        y=Xstd[temp==name,4],
    )
plt.title('Before PCA')
plt.legend( ('died', 'euthanized', 'lived'))

#add PCA ckitlearn shortcut


