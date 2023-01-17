# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

from sklearn.linear_model import LogisticRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

df=pd.read_csv("/kaggle/input/mri-and-alzheimers/oasis_cross-sectional.csv")

df
# df = df[~np.isfinite(df['Delay'])]

df=df.drop(columns=["ID","Delay","Hand"])

df[["CDR"]]=df.CDR.fillna(value=0)!=0
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

catColumns = df.select_dtypes(['object']).columns

for col in catColumns:

    n = len(df[col].unique())

    if (n > 2):

       X = pd.get_dummies(df[col])

       X = X.drop(X.columns[0], axis=1)

       df[X.columns] = X

       df.drop(col, axis=1, inplace=True)  # drop the original categorical variable (optional)

    else:

       le.fit(df[col])

       df[col] = le.transform(df[col])

        

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

df=pd.DataFrame(my_imputer.fit_transform(df),columns=df.columns)
df
lg=LogisticRegression()

df1=df[df.columns[~df.columns.isin(['CDR'])]]

df2=df["CDR"]!=0



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df1,df2, test_size = 0.2, random_state = 42)

lg.fit(x_train,y_train)
np.mean(lg.predict(x_train)!=y_train) # train error


np.mean(lg.predict(x_test)!=y_test) # test error