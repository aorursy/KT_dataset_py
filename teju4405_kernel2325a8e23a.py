# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from nibabel.testing import data_path
import nibabel as nib
from scipy.io import loadmat
import h5py
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
submission=pd.read_csv('/kaggle/input/trends-assessment-prediction/sample_submission.csv')
submission.head(12)
submission.describe()



loading=pd.read_csv('/kaggle/input/trends-assessment-prediction/loading.csv').dropna()
loading.head()
loading.describe()
colm=loading.columns.to_list()[1:]
colm=[i.split(',') for i in colm]
print(np.unique(colm))
img=nib.load('/kaggle/input/trends-assessment-prediction/fMRI_mask.nii')
epi_img_data=img.get_fdata()
img1=img.header
print(img1)

plt.imshow(epi_img_data[:,:,20])
img.affine
print(img1['srow_x'])
print(img1['srow_y'])
print(img1['srow_z'])

def show_slices(slices):
   
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="higher")
slice_0 = epi_img_data[26, :, :]
slice_1 = epi_img_data[:, 25, :]
slice_2 = epi_img_data[:, :, 16]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image") 
train=pd.read_csv('/kaggle/input/trends-assessment-prediction/train_scores.csv').dropna()
train['domain1']=train['domain1_var1']+train['domain1_var2']
train['domain2']=train['domain2_var1']+train['domain2_var2']
train=train.drop(train.iloc[:,2:6],axis=1)
train.head()
train.describe()
sns.scatterplot(train['age'],train['domain1'])
x=train['age'].values.reshape(-1,1)
y=train[['domain1']]
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

lm=LinearRegression()
std=StandardScaler()

lm.fit(x_train,y_train)
lm.predict(x_train)
print('Intercept',lm.intercept_)
print('Coefficent',lm.coef_)



z=lm.predict(x_train)
sns.distplot(z)
plt.show()

plt.scatter(x_train,z)
plt.xlabel(' AVerage Age')
plt.title('Relation in domain1 and age')
plt.show()
sns.scatterplot(train['age'],train['domain2'])
z=train[['domain2']]
x_train,x_test,z_train,z_test=train_test_split(x,z,test_size=0.3,random_state=0)
lm.fit(x_train,z_train)
lm.predict(x_train)
print('Intercept',lm.intercept_)
print('Coefficent',lm.coef_)

q=lm.predict(x_train)
sns.distplot(q)
plt.show()
plt.scatter(x_train,q)
plt.xlabel('Average Age')
plt.title('Relation in domain2 and age')
plt.show()
fnc=pd.read_csv('/kaggle/input/trends-assessment-prediction/fnc.csv')
fnc.head()
colmn=fnc.columns.to_list()[1:]
colmn=[i.split('_')[0] for i in colmn]
print(np.unique(colmn))
icn=pd.read_csv('/kaggle/input/trends-assessment-prediction/ICN_numbers.csv')
icn.head()
icn['ICN_number'].mean()
icn.plot.bar(figsize=(10,7),linewidth=0.4)
merge=pd.merge(train,icn,how='outer',left_index=True,right_index=True).dropna().set_index('Id')
merge.head()
merge['age']=(merge['age']-merge['age'].mean())/merge['age'].std()
merge['domain1']=(merge['domain1']-merge['domain1'].mean())/merge['domain1'].std()
merge['domain2']=(merge['domain2']-merge['domain2'].mean())/merge['domain2'].std()
merge['ICN_number']=(merge['ICN_number']-merge['ICN_number'].mean())/merge['ICN_number'].std()
merge.head()

merge.describe()
merge1=pd.merge(merge,loading,how='inner',left_index=True,right_index=True).dropna()
merge1=merge1.drop(['Id'],axis=1)
merge1.head()
sns.scatterplot(merge['age'],merge['ICN_number'])
w=merge['age'].values.reshape(-1,1)
s=merge[['ICN_number']]
w_train,w_test,s_train,s_test=train_test_split(w,s,test_size=0.3,random_state=0)
lm.fit(w_train,s_train)
lm.predict(w_train)
print('Coefficent',lm.coef_)
print('Intercept',lm.intercept_)
a=lm.predict(w_train)
sns.distplot(a)
plt.scatter(w_train,a)
plt.title('Relation of ICN_number and age')
plt.xlabel('Age')
plt.show()
sns.heatmap(merge.corr(),annot=True)
mat=h5py.File('/kaggle/input/trends-assessment-prediction/fMRI_test/20305.mat','r')
print(mat)
sns.heatmap(merge1.corr(),annot=False,linewidth=-0.1)