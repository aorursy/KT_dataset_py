import numpy as np

import pandas as pd

from sklearn import preprocessing, tree, neighbors

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn import cluster

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



warnings.filterwarnings("ignore")





def handle_non_numeric(df):

    columns = df.columns.values

    for col in columns:

        text_digit_vals = {}

        def convert_to_int(val):

            return text_digit_vals[val]

        

        if df[col].dtype != np.int64 and df[col].dtype != np.float64:

            column_contents = df[col].values.tolist()

            unique_elements = set(column_contents)

            x = 0

            for unique in unique_elements:

                if unique not in text_digit_vals:

                    text_digit_vals[unique] = x

                    x += 1

            df[col] = list(map(convert_to_int,df[col]))

    return df

    



df_o = pd.read_csv("../input/Family Income and Expenditure.csv")



#fill missing values for Household Head Occupation and Household Head Class of Worker with 'unknown'

df_o['Household Head Occupation'] = df_o['Household Head Occupation'].fillna('unknown occupation')

df_o['Household Head Class of Worker'] = df_o['Household Head Class of Worker'].fillna('unknown class')

print('Done.')
corr = df_o.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(200,100))

# plt.tick_params(axis='both', which='major', labelsize=60)

plt.xticks(fontsize=60)

plt.yticks(fontsize=60)

sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light"),square=True)

plt.title("Filipino Household Income Correlation Heatmap",fontsize=100)

plt.show()
df_o = handle_non_numeric(df_o)



X = np.array(df_o.drop('Total Household Income',1))

X = preprocessing.scale(X)

y = np.array(df_o['Total Household Income'])

y = preprocessing.scale(y)

print('Done.')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestRegressor(n_estimators=50)

rf.fit(X_train,y_train)

accuracy = rf.score(X_test,y_test)

print(accuracy)