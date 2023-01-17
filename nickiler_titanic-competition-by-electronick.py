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
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager
%matplotlib inline
titanic_data = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')
titanic_test_data = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')
titanic_data.head()
titanic_data.describe()
fig, ax = plt.subplots()
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#909090'
plt.rcParams['axes.labelcolor']= '#909090'
plt.rcParams['xtick.color'] = '#909090'
plt.rcParams['ytick.color'] = '#909090'
plt.rcParams['font.size']=12

color_palette_list = ['#009ACD', '#ADD8E6', '#63D1F4', '#0EBFE9',   
                      '#C1F0F6', '#0099CC']

survived = titanic_data['Survived'].value_counts(dropna=False)
total = survived[0] + survived[1]
per_survived = survived[1]/total * 100
per_dead = survived[0]/total * 100


labels = ['Survivor', 
         'Deceased']
percentages = [per_survived, per_dead]
explode=(0.05,0)
ax.pie(percentages, explode=explode, labels=labels,  
       colors=color_palette_list[0:2], autopct='%1.1f%%', 
       shadow=False, startangle=-400,   
       pctdistance=1.2,labeldistance=1.5)
ax.axis('equal')
ax.set_title("Surviving people on dataset")
ax.legend(frameon=False, bbox_to_anchor=(1.5,0.8))
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
#import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
#data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Select target
y = titanic_data.Survived

# To keep things simple, we'll use only numerical predictors
titanic_predictors = titanic_data.drop(['Survived'], axis=1)
X = titanic_predictors.select_dtypes(exclude=['object'])

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))



# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
#def score_dataset(X_train, X_valid, y_train, y_valid):

model = RandomForestRegressor(n_estimators=10, random_state=0)
model.fit(imputed_X_train, y_train)
preds = model.predict(imputed_X_valid)

preds.round()
X = titanic_test_data.select_dtypes(exclude=['object'])
imputed_X = pd.DataFrame(my_imputer.fit_transform(X))
imputed_X.columns = X.columns
final_preds = model.predict(imputed_X).round()
len(final_preds)

#pd.DataFrame(titanic_test_data.PassengerId)
titanic_test_data = pd.read_csv('../input/titanic/train.csv')
titanic_test_data.PassengerId

output = pd.DataFrame({'Id': titanic_test_data.PassengerId,'Survived': final_preds})
#output.to_csv('submission.csv', index=False)
output