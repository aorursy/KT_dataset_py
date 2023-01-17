# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline
raw_data = pd.read_csv('../input/Absenteeism_at_work.csv')

raw_data.head()
#Below code it just to let Ipython to show all the columns and rows

pd.options.display.max_columns = None

pd.options.display.max_rows = None
df = raw_data.copy()

df.info() # this will give us number of rows and data types at each column 

#There are no missing values in our table
df.describe()
# we already said there is no missing value another way to do this is to plot heatmap graph df.isnull()

sns.heatmap(df.isnull(),cbar=False, yticklabels=False, cmap='plasma')
# Let's get rid of unnecessary columns

df.drop(['ID'], axis=1, inplace=True)
# Let's how many reasons for absence are there

# We already know from descriptive statistics that min is 0 and max is 28

df['Reason for absence'].unique()
sorted(df['Reason for absence'].unique())
reasons = pd.get_dummies(df['Reason for absence'], drop_first=True)

reasons.head()
reason_type1 = reasons.iloc[:, 0:14].max(axis=1)

reason_type2 = reasons.iloc[:, 15:17].max(axis=1)

reason_type3 = reasons.iloc[:, 18:21].max(axis=1)

reason_type4 = reasons.iloc[:, 22:28].max(axis=1)
reason_type1.head(10)
df = pd.concat([df, reason_type1, reason_type2, reason_type3, reason_type4], axis=1)

df.head()
# drop 'Reason for absence' column

df.drop('Reason for absence', axis=1, inplace=True)
df.columns.values
column_names = ['Month of absence', 'Day of the week',

       'Seasons', 'Transportation expense',

       'Distance from Residence to Work', 'Service time', 'Age',

       'Work load Average/day ', 'Hit target', 'Disciplinary failure',

       'Education', 'Body mass index', 'Absenteeism time in hours', 'reason_1',

       'reason_2', 'reason_3', 'reason_4']
df.columns = column_names

df.head()
df = df[['reason_1', 'reason_2','reason_3', 'reason_4', 'Month of absence', 'Day of the week',

       'Seasons', 'Transportation expense',

       'Distance from Residence to Work', 'Service time', 'Age',

       'Work load Average/day ', 'Hit target', 'Disciplinary failure',

       'Education', 'Body mass index', 'Absenteeism time in hours']]

df.head()
df_reason_modified = df.copy()
df_reason_modified['Education'].unique()
df_reason_modified['Education'].value_counts()
df_reason_modified['Education'] = df_reason_modified['Education'].map({1:0, 2:1, 3:1, 4:1})

df_reason_modified['Education'].unique()
df_reason_modified.drop(['Hit target'], axis=1, inplace=True)
df_preprocessed = df_reason_modified.copy()

df_preprocessed.head(10)
median = df_preprocessed['Absenteeism time in hours'].median()

median
targets = np.where(df_preprocessed['Absenteeism time in hours']>median, 1,0)
targets[:10]
# let's check the ratio

targets.sum()/targets.shape[0]

df_preprocessed['Excessive Absenteeism'] = targets

df_preprocessed.head()
# data_with_targets = df_preprocessed.drop(['Absenteeism time in hours', 'reason_4', 'Body mass index', 'Age', 'Day of the week', 'Distance from Residence to Work'], axis=1)

data_with_targets = df_preprocessed.drop(['Absenteeism time in hours'], axis=1)
data_with_targets.head()
unscaled_inputs = data_with_targets.iloc[:,:-1]

unscaled_inputs.head()
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import StandardScaler



class CustomScaler(BaseEstimator, TransformerMixin):

    def __init__(self, columns, copy=True, with_mean=True, with_std=True):

        self.scaler = StandardScaler(copy, with_mean, with_std)

        self.columns = columns

        self.mean_ = None

        self.std_ = None

    

    def fit(self, X, y=None):

        self.scaler.fit(X[self.columns], y)

        self.mean_ = np.mean(X[self.columns])

        self.std_ = np.std(X[self.columns])

        return self

    

    def transform(self, X, y=None, copy=None):

        init_col_order = X.columns

        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)

        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]

        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]

        
unscaled_inputs.columns.values
columns_to_omit = ['reason_1', 'reason_2', 'reason_3', 'reason_4','Disciplinary failure', 'Education',]

columns_to_scale = [x for x in unscaled_inputs if x not in columns_to_omit]
absenteeism_scaler = CustomScaler(columns_to_scale)
absenteeism_scaler.fit(unscaled_inputs)
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)
scaled_inputs.head()
scaled_inputs.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_inputs, targets, test_size=0.2, 

                                                   random_state=42)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
lr_model = LogisticRegression(solver='lbfgs')

lr_model.fit(X_train, y_train)
lr_model.score(X_train, y_train)
# or we could calculate accuracy manually like this

outputs = lr_model.predict(X_train)
outputs
print('total corretly predicted', np.sum(outputs == y_train))

print('accuracy', np.sum(outputs == y_train)/ outputs.shape[0])
summary_table = pd.DataFrame(columns=['Feature'], data=unscaled_inputs.columns.values)

summary_table['Coefficients'] = lr_model.coef_.T # Takiign the transpose

summary_table
# Let's add intercept as well

summary_table.index = summary_table.index+1

summary_table.loc[0] = ['Intercept', lr_model.intercept_[0]]

summary_table = summary_table.sort_index()

summary_table
summary_table.sort_values('Coefficients', ascending=False)
lr_model.score(X_test, y_test)