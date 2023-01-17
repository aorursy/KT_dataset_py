# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

accidents = pd.read_csv("../input/dft-accident-data/Accidents0515.csv")
accidents.head()
casualties=pd.read_csv('../input/dft-accident-data/Casualties0515.csv' , error_bad_lines=False,index_col='Accident_Index',warn_bad_lines=False)
sns.set_style('whitegrid')
sns.countplot(x='Accident_Severity',data=accidents)
yr = accidents.loc[:,'Date'].groupby(accidents['Date'].map(lambda x: x[6:10])).count()
yr = yr.to_frame()
yr['Year'] = yr.index
yr.columns = ['Accidents','Year']
yr
yr.plot(kind="bar",x="Year",y="Accidents")

sns.set_style('whitegrid')
sns.countplot(x='Day_of_Week',data=accidents)
sns.set_style('whitegrid')
sns.countplot(x='Speed_limit',data=accidents)
#It has only 0 as value so we will delete this column

sns.set_style('whitegrid')
sns.countplot(x='Pedestrian_Crossing-Human_Control', data=accidents)
sns.set_style('whitegrid')
sns.countplot(x='Light_Conditions',hue='Accident_Severity',data=accidents,palette='rainbow')
#Now we will check which column has the missing values 
features_with_na=[features for features in accidents.columns if accidents[features].isnull().sum()>1]

for feature in features_with_na:
    print(feature, np.round(accidents[feature].isnull().mean(), 4),  ' % missing values')
    
    
corr =  accidents.corr()
plt.subplots(figsize=(30,12))
sns.heatmap(corr)

accidents.drop(['Location_Easting_OSGR','Date','Time', 'Location_Northing_OSGR','LSOA_of_Accident_Location','Local_Authority_(Highway)','Accident_Index','Longitude','Latitude','Pedestrian_Crossing-Human_Control'], axis=1, inplace=True)

accidents.head()
model = accidents.drop('Number_of_Casualties',axis=1)
model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
X_train, X_test, y_train, y_test = train_test_split(model.values, 
                                              accidents['Number_of_Casualties'].values,test_size=0.2, random_state=99)
L_R = LogisticRegression()
L_R.fit(X_train,y_train)
Y_pred = L_R.predict(X_test)
L_R.score(X_test, y_test)
acc_L_R1 = round(L_R.score(X_test, y_test) * 100, 2)

sk_report = classification_report(
    digits=6,
    y_true=y_test, 
    y_pred=Y_pred)
print("Accuracy" , acc_L_R1)
print(sk_report)
pd.crosstab(y_test, Y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)