import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot  as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
train = pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')

test = pd.read_csv('/kaggle/input/mobile-price-classification/test.csv')
train.head()
test.head()
train.info()
train.isnull().sum()
train.describe()
doplic = train.duplicated()

doplic.head()
sum(doplic)
numeric_data = train.drop(['blue','dual_sim','four_g','three_g','touch_screen','wifi','price_range'],axis=1) #axis=1 column wise

numeric_data.head()
categorical_data = train[['blue','dual_sim','four_g','three_g','touch_screen','wifi','price_range']]

categorical_data.head()
for col in numeric_data.columns:

    plt.subplots()

    sns.boxplot(numeric_data[col],orient='v')
scaler = StandardScaler()

scaler_array = scaler.fit_transform(numeric_data)
scaled_data = pd.DataFrame(scaler_array,columns=numeric_data.columns)

scaled_data.head()
scaled_data.describe()
plt.subplots(figsize=(10,8))

bp = sns.boxplot(data = scaled_data )

bp.set_xticklabels(bp.get_xticklabels(),rotation=90)
Q1 = scaled_data.quantile(0.25)

Q3 = scaled_data.quantile(0.75)



IQR = Q3 - Q1

print(IQR)
outlier_removed_data = scaled_data[~((scaled_data < (Q1 - 1.5 * IQR)) | (scaled_data > (Q3 + 1.5 * IQR))).any(axis=1)]

outlier_removed_data.shape
plt.subplots(figsize=(10,8))

bp = sns.boxplot(data = outlier_removed_data )

bp.set_xticklabels(bp.get_xticklabels(),rotation=90)
scaled_data = scaled_data.reset_index()

categorical_data =  categorical_data.reset_index()


final_df = pd.concat([scaled_data,categorical_data],axis=1)

final_df = final_df.drop(['index'],axis=1)

final_df.head()
final_df.info()
X = final_df.drop('price_range',axis=1)

Y = final_df['price_range']

X.head()
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.20,random_state=101)
X_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=10000)
logistic_regression.fit(X_train,Y_train)
logistic_regression.score(X_test,Y_test)