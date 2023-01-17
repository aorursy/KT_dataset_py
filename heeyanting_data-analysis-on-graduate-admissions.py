import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')

num_rows = df.shape[0]

num_unique_serial_nos = df['Serial No.'].nunique()

no_duplicate = num_rows == num_unique_serial_nos

print('No duplicate serial numbers:', no_duplicate)
df.drop('Serial No.', axis = 1, inplace = True)

df.head()
df.columns
df.rename(columns={'LOR ': 'LOR', 'Chance of Admit ': 'Chance of Admit'}, inplace = True)

df.columns
print('There is missing data:', df.isnull().values.any())
print('-----Continuous parameters-----')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (10, 10))

sns.kdeplot(

    data = df['GRE Score'],

    kernel = 'gau',

    ax = ax1

)



sns.kdeplot(

    data = df['TOEFL Score'],

    kernel = 'gau',

    ax = ax2

)



sns.kdeplot(

    data = df['CGPA'],

    kernel = 'gau',

    ax = ax3

)



sns.kdeplot(

    data = df['Chance of Admit'],

    kernel = 'gau',

    ax = ax4

)



plt.show()
print('GRE Score has a skewness value of', df['GRE Score'].skew())

print('TOEFL Score has a skewness value of', df['TOEFL Score'].skew())

print('CGPA has a skewness value of', df['CGPA'].skew())

print('Chance of Admit has a skewness value of', df['Chance of Admit'].skew())
print('-----Discrete parameters-----')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (10, 10))

sns.distplot(

    df['University Rating'], ax = ax1,

    bins = 5, kde = False

)



sns.distplot(

    df['SOP'], ax = ax2,

    bins = 9, kde = False

)



sns.distplot(

    df['LOR'], ax = ax3,

    bins = 9, kde = False

)



sns.distplot(

    df['Research'], ax = ax4,

    bins = 2, kde = False

)



plt.show()
corr_mat = df.corr()

mask = np.zeros_like(corr_mat, dtype = np.bool)

mask[np.triu_indices_from(mask)] = True



sns.heatmap(corr_mat, mask = mask, cmap = 'YlGnBu',

           square = True, linewidth = .5, cbar_kws = {'shrink': .5})



plt.show()
plt.figure(figsize = (10, 20))

index = 0



for col in df.columns:

    if col == 'Chance of Admit':

        pass

    else:

        index += 1

        plt.subplot(4, 2, index)

        sns.regplot(

            x = df['Chance of Admit'],

            y = df[col]

        )



plt.show()
X = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']]

y = df['Chance of Admit']
import statsmodels.api as sm

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size = 0.20, random_state = 0)



print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
from sklearn.linear_model import LinearRegression



lm = LinearRegression()

trained_model = lm.fit(X_train, y_train)

predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)

plt.title('Comparison of Predictions with y_test')

plt.xlabel('True Values')

plt.ylabel('Predictions')

plt.show()
from sklearn.metrics import r2_score



print('Variance:', r2_score(y_test, predictions))