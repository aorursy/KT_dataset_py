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
# data processing, CSV file I/O (e.g. pd.read_csv)



train=pd.read_csv('../input/medical-data/train.csv')

#train.info()

#target=train['Stay']

#train=train.drop(['Stay'],axis=1)

test=pd.read_csv('../input/medical-data/test.csv')

#print(train)

#print(test)
#train['Department']

#test=pd.read_csv('../input/medical-data/test.csv')

train
'''dept_list = train['Department'].unique()

dept_list.sort()

dept_dict = dict(zip(dept_list, range(len(dept_list))))

train['Department'].replace(dept_dict, inplace=True)

train['Department']





'''



a=train['City_Code_Patient'].isnull().sum()

b=train['Bed Grade'].isnull().sum()

print(b)

print(a)



#i#=''

#i.null()

#c=zip(a,b)

#for i,j in c:

#    print(i.null(),j)

#print(a.isnull().sum())

#print(b.isnull().sum())

#a=train['City_Code_Patient'].isnull() and train['Bed Grade'].isnull()



train=train.drop(train[(train['City_Code_Patient'].isnull())|(train['Bed Grade'].isnull())].index,axis=0)

#a=aa.index

#aa=aa.drop(,axis=0)

#print(aa.index)

train
for i in train.columns:

    print(train[i].unique())

#len(train['patientid'].unique())



#drop patientid as it doesn't contain usefull information
import seaborn as sns 

#fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize=(6,3))

#ax1.hist(train.Time_Hr[df.Class==0],bins=48,color='g',alpha=0.5)

#ax1.set_title('Genuine')

sns.pairplot(train.iloc[:,:2])

'''

from yellowbrick.target import ClassBalance 

import matplotlib.pyplot as plt 

plt.style.use("ggplot")

plt.rcParams['figure.figsize'] = (8,8)





visualizer=ClassBalance().fit(train.Stay)

visualizer.show()

'''
#train=train.drop(['Bed Grade'],axis=1)

#target

#We have drop the columns so don't need to do this step 

#If want to do this step then undo the previous step where we have 

#drop aall the nan values



'''

def fill_null(df):

  train['Bed Grade'].fillna(train['Bed Grade'].value_counts().index[0],inplace=True)

  train['City_Code_Patient'].fillna(train['City_Code_Patient'].value_counts().index[0],inplace=True)

'''
fill_null(train)

fill_null(test)

#It is like LabelEncodung of a column

def convert_to_numerical(train):

  dept_list = train['Department'].unique()

  dept_list.sort()

  dept_dict = dict(zip(dept_list, range(len(dept_list))))

  train['Department'].replace(dept_dict, inplace=True)





  hrc_list = train['Hospital_region_code'].unique()

  hrc_list.sort()

  hrc_dict = dict(zip(hrc_list, range(len(hrc_list))))

  train['Hospital_region_code'].replace(hrc_dict, inplace=True)



  ward_list = train['Ward_Type'].unique()

  ward_list.sort()

  ward_dict = dict(zip(ward_list, range(len(ward_list))))

  train['Ward_Type'].replace(ward_dict, inplace=True)



  wfc_list = train['Ward_Facility_Code'].unique()

  wfc_list.sort()

  wfc_dict = dict(zip(wfc_list, range(len(wfc_list))))

  train['Ward_Facility_Code'].replace(wfc_dict, inplace=True)



  toa_list = train['Type of Admission'].unique()

  toa_list.sort()

  toa_dict = dict(zip(toa_list, range(len(toa_list))))

  train['Type of Admission'].replace(toa_dict, inplace=True)



  soi_list = train['Severity of Illness'].unique()

  soi_list.sort()

  soi_dict = dict(zip(soi_list, range(len(soi_list))))

  train['Severity of Illness'].replace(soi_dict, inplace=True)



  age_list = train['Age'].unique() 

  age_list.sort()

  age_dict = dict(zip(age_list, range(len(age_list))))

  train['Age'].replace(age_dict, inplace=True)



  htc_list = train['Hospital_type_code'].unique()

  htc_list.sort()

  htc_dict = dict(zip(htc_list, range(len(htc_list))))

  train['Hospital_type_code'].replace(htc_dict, inplace=True)

convert_to_numerical(train)

convert_to_numerical(test)
#As we have done label encoding above so don.t need to calculate further 



'''

#train['Age'].unique()

from sklearn.preprocessing import LabelEncoder

lb=LabelEncoder()

for i in train.columns:

    train[i]=lb.fit_transform(train[i])

    test[i]=lb.transform(test[i])

#train=pd.get_dummies(train,drop_first=True)

#test=pd.get_dummies(test,drop_first=True)



'''
#train.iloc[:,5]
#As we have completed fill null function above so don't need to complete using Imputer 

'''

#train.info()

#print(train)

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer(strategy='mean')

train_new = my_imputer.fit_transform(train.iloc[:,5:6])

test_new = my_imputer.fit_transform(test.iloc[:,5:6])

'''
#train['City_Code_Patient']=train_new

#test['City_Code_Patient']=test_new

#test_new=test

target=train['Stay']

train=train.drop(['Stay'],axis=1)
from sklearn.preprocessing import LabelEncoder

lb=LabelEncoder()

target=lb.fit_transform(target)
train=train.drop(['case_id','patientid'],axis=1)
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(train,target,test_size=0.2)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(X_train)

X_test=sc.transform(X_test)

#test_new=sc.transform(test_new)
#from sklearn.preprocessing import Normalizer

#nm=Normalizer()

#X_train=nm.fit_transform(X_train)

#X_test=nm.transform(X_test)
#from sklearn.decomposition import PCA

#pvc=PCA(n_components=1)

#X_train=pvc.fit_transform(X_train)

#X_test=pvc.transform(X_test)

#X_test
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#ld=LDA(n_components=3)

#X_train=ld.fit_transform(X_train,y_train)

#X_test=ld.transform(X_test)

#test_new=ld.transform(test_new)
#from sklearn.linear_model import LinearRegression

#from math import sqrt

#from sklearn.metrics import mean_squared_error

#rc=LinearRegression()

#rc.fit(X_train,y_train)

#predic=rc.predict(X_test)

#print(sqrt(mean_squared_error(y_test,predic)))
import xgboost

from math import sqrt

from sklearn.metrics import confusion_matrix

#from sklearn.metrics import confusion_matrix

clf = xgboost.XGBClassifier()

clf.fit(X_train,y_train)

y_testpred= clf.predict(X_test)



rms = confusion_matrix(y_test, y_testpred)

print("RMSE:", rms)

#y_pred = clf.predict(test)
caseid=test['case_id']

test=test.drop(['case_id','patientid'],axis=1)

test=sc.transform(test)
from sklearn.metrics import accuracy_score



print(accuracy_score(y_test, y_testpred))



prediction=clf.predict(test)

prediction=lb.inverse_transform(prediction)

output = pd.DataFrame({'case_id': caseid, 'Stay': prediction})

output.to_csv('New_Submission.csv', index=False)