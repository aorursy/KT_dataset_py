# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
test = pd.read_csv("../input/test.csv",index_col="Id")

train = pd.read_csv("../input/train.csv",index_col="Id")

train.head()
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))
all_data = pd.get_dummies(all_data)
plt.figure(figsize=(12,5))

prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})

prices.hist()
#log transform the target:

train["SalePrice"] = np.log1p(train["SalePrice"])



#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
#filling NA's with the mean of the column:

all_data = all_data.fillna(all_data.mean())
all_data.tail()
house_train = all_data[:train.shape[0]]

#house_test = all_data[train.shape[0]:]

house_prices = train.SalePrice
from sklearn.model_selection import train_test_split, KFold

from sklearn.ensemble import RandomForestRegressor
X_train, X_test, y_train, y_test = train_test_split(

    house_train, house_prices, test_size=0.4, random_state=0)
print(X_train.shape,X_test.shape)
rf = RandomForestRegressor()

rf = rf.fit(X_train,y_train)



#y_pred = rf.predict(X_test)

print(rf.score(X_test,y_test))
number_params = 15

ix = np.argsort(rf.feature_importances_)[::-1]

index = np.arange(number_params)

bar_width = 0.35

fig, ax = plt.subplots(figsize=(5,5))

ax.bar(index,rf.feature_importances_[ix][:number_params],bar_width)

ax.set_ylabel("parameter importance")

ax.set_xticks(index + bar_width)

ax.set_xticklabels(X_train.columns.values[ix][:number_params],rotation=60)
from sklearn.decomposition import PCA



pca = PCA(n_components=20)

pca.fit(X_train)

X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)
rf = RandomForestRegressor()

rf = rf.fit(X_train_pca,y_train)



print(rf.score(X_test_pca,y_test))
plt.plot(np.sort(rf.feature_importances_)[::-1],'-o')

#plt.xlim(0,40)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
test = pd.read_csv("../input/test.csv",index_col="Id")

train = pd.read_csv("../input/train.csv",index_col="Id")

train.head()
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))
all_data = pd.get_dummies(all_data)
#filling NA's with the mean of the column:

all_data = all_data.fillna(all_data.mean())
all_data.tail()
house_train = all_data[:train.shape[0]]

#house_test = all_data[train.shape[0]:]

house_prices = np.log1p(train.SalePrice)