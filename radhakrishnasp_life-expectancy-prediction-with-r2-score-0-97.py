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
data = pd.read_csv('../input/life-expectancy-who/Life Expectancy Data.csv')
data.head()
data.shape
data.info()
data.isnull().sum()
def missing_values(df):

    missing=pd.DataFrame(df.isnull().sum()/len(data))*100

    missing.columns = ['missing_values(%)']

    missing['missing_values(numbers)'] = pd.DataFrame(df.isnull().sum())

    return missing.sort_values(by='missing_values(%)', ascending=False)

missing_values(data)
# Renaming some column names as they contain trailing spaces.

data.rename(columns={" BMI ":"BMI","Life expectancy ":"Life_Expectancy","Adult Mortality":"Adult_Mortality",

                   "infant deaths":"Infant_Deaths","percentage expenditure":"Percentage_Exp","Hepatitis B":"HepatitisB",

                  "Measles ":"Measles","under-five deaths ":"Under_Five_Deaths","Diphtheria ":"Diphtheria",

                  " HIV/AIDS":"HIV/AIDS"," thinness  1-19 years":"thinness_1to19_years"," thinness 5-9 years":"thinness_5to9_years","Income composition of resources":"Income_Comp_Of_Resources",

                   "Total expenditure":"Tot_Exp"},inplace=True)
data.head()
for label,content in data.items():

    if pd.isnull(content).sum():

        data[label] = content.fillna(content.median())
missing_values(data)
data=pd.get_dummies(data, columns=['Country','Status'])
data.head()
X = data.drop('Life_Expectancy', axis=1)

y = data['Life_Expectancy']
X.head()
y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()

gbr.fit(X_train, y_train)

gbr_pred = gbr.predict(X_test)

print('R2 score is : {:.2f}'.format(r2_score(y_test, gbr_pred)))
df = data.copy()
Q1 = df.quantile(0.25)

Q3 = df.quantile(0.75)

IQR = Q3 - Q1

outliers = pd.DataFrame(((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum())

outliers1= outliers[:60]

outliers2 = outliers[60:120]

outliers3 = outliers[120:180]

outliers4 = outliers[180:]

outliers1,outliers2,outliers3,outliers4
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred=rf.predict(X_test)
print('R2 score is : {:.2f}'.format(r2_score(y_test, rf_pred)))
rf.feature_importances_
import seaborn as sns

# Helper function for plotting feature importance

def plot_features(columns, importances, n=10):

    df = (pd.DataFrame({"features": columns,

                        "feature_importance": importances})

          .sort_values("feature_importance", ascending=False)

          .reset_index(drop=True))

    

    sns.barplot(x="feature_importance",

                y="features",

                data=df[:n],

                orient="h")
plot_features(X_train.columns, rf.feature_importances_)
new_data = data[['HIV/AIDS','Adult_Mortality','Income_Comp_Of_Resources','Schooling',

      'BMI','thinness_5to9_years','Under_Five_Deaths','Infant_Deaths',

      'thinness_1to19_years','Year']]
new_data
X_train, X_test, y_train, y_test = train_test_split(new_data,y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
rf.fit(X_train, y_train)
rf_pred_new = rf.predict(X_test)
print('R2 score is : {:.2f}'.format(r2_score(y_test, rf_pred_new)))
rf_pred_new = pd.DataFrame(rf_pred_new)
rf_pred_new.to_csv('predictions.csv')