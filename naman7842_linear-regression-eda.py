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
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df = pd.read_csv("/kaggle/input/kc-housesales-data/kc_house_data.csv")
print(df.shape)

df.info()

df.describe()
df.head()
#dropping date and id as they won't affect the prcie prediction

df.drop(['date','id'], inplace = True, axis=1)
df['age'] = 2020-df['yr_built']
df.drop('yr_built', axis=1, inplace=True)
import seaborn as sns

sns.distplot(df['price'])

print("skewness :", df['price'].skew())

print("kurtosis :",df['price'].kurt())
df_boxplot = df[['sqft_living', 'sqft_above', 'sqft_basement', 'sqft_living15', 'age']]

df_barplot = df[['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'grade']]
for i in df_boxplot.columns:

    #sns.set(style='white')

    plt.figure(figsize=(15,5))

    #print("boxplot of %s" %(i))

    sns.boxplot(x=i, data=df)

    plt.show()
for i in df_barplot.columns:

    plt.figure(figsize=(10,5))

    cat_num = df[i].value_counts()

    sns.barplot(x=cat_num.index, y=cat_num)

    plt.show()
sns.set()

plt.figure(figsize=(20,10))

sns.heatmap(df.corr(),annot=True)
df.drop(['sqft_lot','condition','yr_renovated','zipcode','long','sqft_lot15'],axis=1, inplace=True)
X = df.drop('price',1)

y=df['price']

y = np.log(y) #since the price distribution is positively skewed, thus, doing logarithmic transformation
X
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train,y_train)

predictions = model.predict(X_test)
sns.set_style("whitegrid")

sns.regplot(y_test,predictions)
from sklearn.metrics import r2_score

print("score : ",r2_score(y_test,predictions))