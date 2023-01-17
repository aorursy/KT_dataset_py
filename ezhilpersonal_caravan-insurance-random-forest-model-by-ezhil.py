import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import OneHotEncoder



from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
pd.options.display.max_columns=100

pd.options.display.max_rows=100
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
df = pd.read_csv("../input/caravan-insurance-challenge/caravan-insurance-challenge.csv")
df.head()
df.shape
df.info()
df.isnull().values.any()
df.nunique()
x_train = df.loc[df['ORIGIN'] == 'train',:]

x_test = df.loc[df['ORIGIN'] == 'test',:]
a = x_train.pop('ORIGIN')

b = x_test.pop('ORIGIN')

y_train = x_train.pop('CARAVAN')

y_test = x_test.pop('CARAVAN')
x_train.shape

x_test.shape

y_train.shape

y_test.shape
num_columns = x_train.columns[x_train.nunique() > 5]

cat_columns = x_train.columns[x_train.nunique() <= 5]
len(num_columns)

len(cat_columns)
num_columns
plt.figure(figsize=(15,15))

sns.distributions._has_statsmodels=False

for i in range(len(num_columns)):

    plt.subplot(11,5,i+1)

    sns.distplot(df[num_columns[i]])

    

plt.tight_layout()
ct = ColumnTransformer(

                        [

                            ('num_col', RobustScaler(),num_columns), 

                            ('cat_col', OneHotEncoder(handle_unknown='ignore'), cat_columns),

                         ]

,remainder = 'passthrough')
ct.fit_transform(x_train) 
pipe = Pipeline([('ct',ct),

                 ('rf',RandomForestClassifier())

                 ])

pipe.fit(x_train,y_train)
y_pred = pipe.predict(x_test)
np.sum(y_pred == y_test)/len(y_test)*100