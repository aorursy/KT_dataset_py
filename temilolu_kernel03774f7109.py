# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as DT
from datetime import datetime

import time
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier


# Path of the file to read
train_file_path = '../input/train_technidus_clf.csv'
test_file_path = '../input/test_technidus_clf.csv'

#Variable to access data
train_data=pd.read_csv(train_file_path)
test_data=pd.read_csv(test_file_path)

#Describe the data
#print(train_data.describe())
train_data.shape
print('Test Data')
print(test_data.shape)

print('Train Data')
print(train_data.shape)
#Check and print columns with missing values
missing_val_count_by_column = (train_data.isnull().sum())
#print(missing_val_count_by_column [missing_val_count_by_column > 0])
 
train_data.TotalChildren=train_data.TotalChildren.fillna(train_data.TotalChildren.mode())
train_data.TotalChildren=train_data.TotalChildren.fillna(train_data.HomeOwnerFlag.mode())

missing_val_count_by_column2 = (train_data.isnull().sum())
#print(missing_val_count_by_column2 [missing_val_count_by_column2 > 0])
# Any results you write to the current directory are saved as output.


#Convert Education column to int
def Education_to_numeric(bp):
    if bp == 'Bachelors ':
        return 4
    if bp == 'Partial College':
        return 3
    if bp == 'High School':
        return 2
    if bp == 'Graduate Degree':
        return 1
    if bp == 'Partial High School':
        return 0
edu=train_data.Education
edut=test_data.Education
edupp = edu.apply(Education_to_numeric)
edud = edut.apply(Education_to_numeric)
train_data.Education=edupp
test_data.Education=edud



#Convert Occupation column to int
def occupation_to_numeric(b):
    if b == 'Clerical':
        return 1
    if b == 'Professional':
        return 1
    if b == 'Manual':
        return 0
    if b == 'Skilled Manual':
        return 0
    if b == 'Management':
        return 1
p=train_data.Occupation
pt=test_data.Occupation
d = p.apply(occupation_to_numeric)
dt = pt.apply(occupation_to_numeric)
train_data.Occupation=d
test_data.Occupation=dt



#Convert Occupation column to int
def COUNTRY_to_numeric(d):
    if d == 'Canada':
        return 3
    if d == 'France':
        return 4
    if d == 'Australia':
        return 1
    if d == 'United Kingdom':
        return 2
    if d == 'United States':
        return 5
    if d == 'Germany':
        return 6

pon=train_data.CountryRegionName
ptu=test_data.CountryRegionName
qa = pon.apply(COUNTRY_to_numeric)
qaw= ptu.apply(COUNTRY_to_numeric)
train_data.CountryRegionName=qa
test_data.CountryRegionName=qaw


#Convert MaritalStatus column to int
def marital_to_numeric(ip):
    if ip == 'M':
        return 0
    if ip == 'S':
        return 1
pl= train_data.MaritalStatus
plt=test_data.MaritalStatus
ml = pl.apply(marital_to_numeric)
mlt = plt.apply(marital_to_numeric)
train_data.MaritalStatus=ml
test_data.MaritalStatus=mlt

#Convert Gender column to int
#def car_to_numeric(car):
 #   if car ==0:
  #      return 2
   # if car >=1:
    #    return 1
#iz=train_data.NumberCarsOwned
#izt=test_data.NumberCarsOwned
#ci = iz.apply(car_to_numeric)
#cti = izt.apply(car_to_numeric)
#train_data.NumberCarsOwned=ci
#test_data.NumberCarsOwned=cti


#Convert Gender column to int
def region_to_numeric(a):
    if a == 'M':
        return 2
    if a == 'F':
        return 1
z=train_data.Gender
zt=test_data.Gender
c = z.apply(region_to_numeric)
ct = zt.apply(region_to_numeric)
train_data.Gender=c
test_data.Gender=ct


#Convert train Birthdate column to age(int)
yy=train_data.BirthDate
now=pd.Timestamp(DT.datetime.now())
nuu = pd.to_datetime(yy)
nuu=(now-nuu)/365
ny=nuu.dt.days
train_data.BirthDate=ny


#Convert test Birthdate column to age(int)
yyt=test_data.BirthDate
nowt=pd.Timestamp(DT.datetime.now())
nuut = pd.to_datetime(yyt)
nuut=(nowt-nuut)/365
nyt=nuut.dt.days
test_data.BirthDate=nyt

#childrennithome
#df["ColC"] = df["ColA"].subtract(df["ColB"], fill_value=0)
train_data.PostalCode=train_data.TotalChildren.subtract(train_data.NumberChildrenAtHome, fill_value=0)
test_data.PostalCode=test_data.TotalChildren.subtract(test_data.NumberChildrenAtHome, fill_value=0)

#income per children
gh=train_data.YearlyIncome-train_data.TotalChildren
#train_data.Suffix=gh


#train_data.YearlyIncome=np.exp(train_data.YearlyIncome)
#test_data.YearlyIncome=np.exp(test_data.YearlyIncome)

#Convert Occupation column to string
def age_to_numeric(qa):
    if qa <=37:
        return 3
    if 51>qa >=44:
        return 4
    if 58>qa >=51:
        return 1
    if 65>qa >=58:
        return 2
    if qa >=65:
        return 5
train_data.BirthDate=100/train_data.BirthDate**3.9
test_data.BirthDate=100/test_data.BirthDate**3.9
#di = pti.apply(age_to_numeric)
#dti= ptit.apply(age_to_numeric)
#train_data.BirthDate=di
#test_data.BirthDate=dti


def ave_to_numeric(qa):
    if qa <40:
        return 1
    if 50>qa >=40:
        return 2
    if 59>qa >=50:
        return 3
    if 65>qa >=59:
        return 4
    if qa >=65:
        return 5
#lo=train_data.AveMonthSpend*100
#lot=test_data.AveMonthSpend*100
#dp = lo.apply(ave_to_numeric)
#dtp= lot.apply(ave_to_numeric)
#train_data.AveMonthSpend=lo
#test_data.AveMonthSpend=lot
train_data.AveMonthSpend=(np.log(train_data.AveMonthSpend))*12
test_data.AveMonthSpend=(np.log(test_data.AveMonthSpend))*12

train_data.Suffix=(train_data.YearlyIncome**3.3)-(train_data.AveMonthSpend**3)
test_data.Suffix=(test_data.YearlyIncome**3.3)-(test_data.AveMonthSpend**3)

train_data.YearlyIncome=(train_data.YearlyIncome**3.3)
test_data.YearlyIncome=(test_data.YearlyIncome**3.3)

#test_data.AveMonthSpend=(1/test_data.AveMonthSpend**4)


train_data.TotalChildren=(train_data.TotalChildren*2)
test_data.TotalChildren=(test_data.TotalChildren*2)

train_data.City=train_data.AveMonthSpend*30
test_data.City=test_data.AveMonthSpend*30

train_data.PostalCode=train_data.TotalChildren.subtract(train_data.NumberChildrenAtHome, fill_value=0)
test_data.PostalCode=test_data.TotalChildren.subtract(test_data.NumberChildrenAtHome, fill_value=0)

train_data.PostalCode=1/train_data.CustomerID**1/2
test_data.PostalCode=1/test_data.CustomerID**1/2

#Let's  Create target object and call it y
y = train_data.BikeBuyer
#train_data['IncomePerchild']=(train_data.YearlyIncome/train_data.TotalChildren).astype(int)
#test_data['IncomePerchild']=(test_data.YearlyIncome/test_data.TotalChildren).astype(int)


# Create X with the required features
features = ['TotalChildren','YearlyIncome',
            'HomeOwnerFlag','NumberChildrenAtHome','Gender','MaritalStatus','BirthDate','CountryRegionName',
            'Occupation','PostalCode','Suffix','Education','PostalCode'
            ]
import matplotlib.pyplot as plt
train_data.AveMonthSpend.plot.hist(color='blue',bins=50)
plt.show()
X= train_data[features]
textX=test_data[features]

import seaborn as sns
corr= train_data.corr()
#corr
f, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(corr,cmap='coolwarm',linewidths=2.0, annot=True)

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
X = my_imputer.fit_transform(X)

textX = my_imputer.fit_transform(textX)




# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


xgb = XGBClassifier(n_estimators=380, learning_rate=0.049, random_state=1,min_child_weight=6)
training_start = time.perf_counter()
xgb.fit(train_X, train_y)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = xgb.predict(val_X)
prediction_end = time.perf_counter()
acc_xgb = (preds == val_y).sum().astype(float) / len(preds)*100
xgb_train_time = training_end-training_start
xgb_prediction_time = prediction_end-prediction_start
print("XGBoost's prediction accuracy is: %3.2f" % (acc_xgb))
print("Time consumed for training: %4.3f" % (xgb_train_time))
print("Time consumed for prediction: %6.5f seconds" % (xgb_prediction_time))


  
pre=xgb.predict(textX)
final=pre.astype(int)
output = pd.DataFrame({'CustomerID': test_data.CustomerID,
                      'BikeBuyer': final})


output.to_csv('Sample Submission file clf .csv', index=False)





