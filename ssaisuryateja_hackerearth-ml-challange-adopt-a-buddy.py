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
train_data = pd.read_csv('../input/hackerearth-ml-challenge-pet-adoption/train.csv')
test_data = pd.read_csv('../input/hackerearth-ml-challenge-pet-adoption/test.csv')

train_data = train_data.drop(['height(cm)'],axis=1)
test_data = test_data.drop(['height(cm)'],axis=1)
train_data = train_data.drop(['length(m)'],axis=1)
test_data = test_data.drop(['length(m)'],axis=1)
foo = test_data#dummy
train_data.head(5)
f_data = train_data.drop(['breed_category','pet_category'],axis=1)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

f_data["condition"] = imputer.fit_transform(f_data[["condition"]]).ravel()
test_data["condition"] = imputer.fit_transform(test_data[["condition"]]).ravel()
test_data.head(5)

#Label Encoding the data for categorical features
from sklearn.preprocessing import LabelEncoder

new_train_data = f_data.apply(LabelEncoder().fit_transform)
new_test_data1 = test_data.apply(LabelEncoder().fit_transform)
new_train_data.head(5)
#Standarization of data
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
scalar.fit(new_train_data)
new_train_data = scalar.transform(new_train_data)
new_train_data = pd.DataFrame(new_train_data)

scalar.fit(new_test_data1)
new_test_data1 = scalar.transform(new_test_data1)
new_test_data1 = pd.DataFrame(new_test_data1)

x_train = new_train_data
x_test = new_test_data1
yb_train = train_data['pet_category']
ya_train = train_data['breed_category']


x_train.head(5)
from sklearn.ensemble import RandomForestClassifier


clf = RandomForestClassifier(n_estimators=175,max_features='log2',bootstrap=True,max_depth=17,min_samples_leaf=2,min_samples_split=3,n_jobs=-1,random_state=3)
clf.fit(x_train,ya_train)
y_predict1 = clf.predict(x_test)
y_predict1 = pd.DataFrame(y_predict1)   #prediction for the breed_category


clf.fit(x_train,yb_train)
y_predict2 = clf.predict(x_test)
y_predict2 = pd.DataFrame(y_predict2)
y_predict2.head()                      #prediction for the pet_category
final_df = foo.drop(['issue_date','listing_date','condition','color_type','X1','X2'],axis=1)
final_df['breed_category'] = y_predict1
final_df['pet_category'] = y_predict2
final_df.to_csv("./file.csv", sep=',',index=True)
final_df.head