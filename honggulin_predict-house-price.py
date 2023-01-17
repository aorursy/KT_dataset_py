import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re #Regular expression operations¶

import matplotlib.pyplot as plt #plot

import seaborn as sns #statistical data visualization¶

from sklearn.preprocessing import LabelEncoder
train = pd.read_csv("../input/train.csv")

train.head(6)
test = pd.read_csv("../input/test.csv")

test.head(6)
sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(figsize=(20,10))

sns.despine(left=True)

d = train['SalePrice']

sns.distplot(d,color='m')

plt.setp(axes, yticks=[])

plt.tight_layout()
train.dtypes
train.shape
test.shape
test_ID = test['Id']
train = train.drop('Id',axis=1)

test = test.drop('Id',axis=1)
print(train.shape)

print(test.shape)
ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.values[:,79]

train = train.drop('SalePrice',axis=1)

data = pd.concat((train, test)).reset_index(drop=True)

#data.drop(['SalePrice'], axis=1, inplace=True)

print("data size is : {}".format(data.shape))
data.head(6)
data_na = (data.isnull().sum() / len(data)) * 100

data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :data_na})

missing_data.head(10)
#Correlation map to see how features are correlated with SalePrice

corrmat = train.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
#fill the NA with "None"

for na in data.isnull():

    data[na] = data[na].fillna("None")
data.dtypes
data.head(6)
#col = list(train.columns.values)
int_col = list(data.select_dtypes(include=[np.number]).columns.values)
col = list(set(list(train.columns.values))-set(int_col))
print(col)
for c in col:

    en = LabelEncoder() 

    en.fit(list(data[c].values)) 

    data[c] = en.transform(list(data[c].values))
data.head(6)
data.dtypes
X_train = data.values[0:1460,0:80]

'''

# define base model

def baseline_model():

    # create model

    model = Sequential()

    model.add(Dense(79, input_dim=79, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

'''

print(X_train.shape)

print(type(X_train))

print(y_train.shape)

print(type(y_train))
X_test = data.values[1460:,0:80]

print(X_test.shape)
from sklearn.linear_model import LassoLars

reg = LassoLars(alpha=.1)

reg.fit(X_train,y_train)

predictions = reg.predict(X_test)
i=0

for x in predictions:

    if x<0:

        predictions[i]=-x

    i=i+1

        
print(predictions.shape)
sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(figsize=(20,10))

sns.despine(left=True)

d = predictions

sns.distplot(d,color='m')

plt.setp(axes, yticks=[])

plt.tight_layout()
# Generate Submission File 

output = pd.DataFrame({ 'Id': test_ID, 'SalePrice': predictions })

output.to_csv("output.csv", index=False)