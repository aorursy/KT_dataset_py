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
import pandas as pd
import numpy as np
df = pd.read_csv('/kaggle/input/dataset-for-svm/Q2_data_set - Q2_data_set.csv')
df.shape
list(df)
df.head()
df.shape[1]
df.COMPANY_CLASS.unique()
df.COMPANY_STATUS.unique()
df.COMPANY_CATEGORY.unique()
df.AUTHORIZED_CAPITAL.unique()
df.REGISTRAR_OF_COMPANIES.unique()
df.PRINCIPAL_BUSINESS_ACTIVITY.unique()
df.SUB_CATEGORY.unique()
df.DATE_OF_REGISTRATION.unique()
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
y=df['PAIDUP_CAPITAL'].astype(str)
##df['COMPANY_STATUS']=LabelBinarizer().fit_transform(df.COMPANY_STATUS)
#df['COMPANY_CLASS']=LabelBinarizer().fit_transform(df.COMPANY_CLASS)
#df['COMPANY_CATEGORY']=LabelBinarizer().fit_transform(df.COMPANY_CATEGORY)
#df['AUTHORIZED_CAPITAL']=LabelBinarizer().fit_transform(df.AUTHORIZED_CAPITAL)
#df['REGISTRAR_OF_COMPANIES']=LabelBinarizer().fit_transform(df.REGISTRAR_OF_COMPANIES)
##f=LabelBinarizer().fit_transform(df.PRINCIPAL_BUSINESS_ACTIVITY)
features=['COMPANY_STATUS','COMPANY_CLASS','COMPANY_CATEGORY','AUTHORIZED_CAPITAL','REGISTRAR_OF_COMPANIES']
x=df[features].astype(str)
# ordinal encode input variables
ordinal_encoder = OrdinalEncoder()
x = ordinal_encoder.fit_transform(x)
# ordinal encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=4000, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
from sklearn import metrics
#Evaluating the algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)
from sklearn.linear_model import LogisticRegression  
classifier= LogisticRegression(random_state=47)  
classifier.fit(x_train, y_train)
y_pred= classifier.predict(x_test) 
from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(y_test, y_pred)) 
