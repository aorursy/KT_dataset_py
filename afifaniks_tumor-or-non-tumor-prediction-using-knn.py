import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
df = pd.read_csv('../input/brain-tumor/bt_dataset_t3.csv')

df.replace(to_replace='None', value=pd.np.nan, inplace=True)

pd.options.display.max_columns=None

pd.options.display.max_rows=None

print(df.shape)

df.head()
y = df['Target']

X = df.drop(['Image', 'Target'], axis=1)

X = X.replace([np.inf, -np.inf], np.nan) # Replacing inf values with NaN
def process_na_columns(df, tolerance=0.3):

    '''

    df: dataset except the target

    tolerance: We drop a column if it has lower valid data than tolerance 

    '''

    dataset_size = df.shape[0]

    tolerate_na_size = int(dataset_size * tolerance)  

    for c in df.columns:

        na_counts = df[c].isna().sum()

        

        if na_counts != 0:

            if na_counts > tolerate_na_size:

                df = df.drop([c], axis=1)

            else:

                if df[c].dtypes == 'object':

                    values = df[c].value_counts()

                    max_val = values.index[0] # Highest Occurrence

                    df[c].replace(to_replace=pd.np.nan, value=max_val, inplace=True)

                else:

                    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

                    data = df[c].values.reshape(-1, 1)

                    imputed_values = imputer.fit_transform(data)

                    df[c] = imputed_values

    return df
X = process_na_columns(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

score = knn.score(X_test, y_test)
print("Score:", score*100, "%")