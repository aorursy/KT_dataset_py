#import some necessary librairies



import numpy as np

import pandas as pd

pd.set_option('display.float_format', lambda x: '%.3f' % x)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style='darkgrid', context='notebook', palette='viridis')

sns.despine(top=True,right=True)



from scipy import stats

from scipy.stats import norm, skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.linear_model import Lasso

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler,StandardScaler,LabelEncoder

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings("ignore")
# Importing and putting the train and test datasets in a pandas dataframe

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# Checking the size of the data.

print ("Train data shape:", train.shape)

print ("Test data shape:", test.shape)
train.head(5)
# Renaming the 'data-price' column to a new column 'Price' 

train['price']=train['data-price']

train.drop("data-price", axis = 1, inplace = True)

train.head(5)
test.head(5)
# Save the 'house-Id' column

train_ID = train['house-id']

house_ID = test['house-id']



# Dropping the 'house-id' and 'data-url' colums since they are unnecessary for the prediction process.

train.drop("house-id", axis = 1, inplace = True)

test.drop("house-id", axis = 1, inplace = True)

train.drop("data-url", axis = 1, inplace = True)

test.drop("data-url", axis = 1, inplace = True)

train.drop("data-date", axis = 1, inplace = True)

test.drop("data-date", axis = 1, inplace = True)
train.head(5)
test.head(5)
# Checking the data types of the test set using info()

test.info()
# Checking the data types of the train set using info()

train.info()
# Checking if our target variable is normally skewed

def checkskew(col):

    sns.distplot(train[col],fit=norm)

    (mu, sigma) = norm.fit(train[col])

    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

checkskew('price')
train['price'] = np.log1p(train['price'])

checkskew('price')
fig, ax = plt.subplots()

ax.scatter(x = train['buildingSize'], y = train['price'])

plt.ylabel('price', fontsize=13)

plt.xlabel('buildingSize', fontsize=13)

plt.show()
# Dropping variables seen as outliers from the data 

train = train.drop(train[(train['buildingSize']>680) & (train['price']<3000000)].index)



fig, ax = plt.subplots()

ax.scatter(train['buildingSize'], train['price'])

plt.ylabel('price', fontsize=13)

plt.xlabel('buildingSize', fontsize=13)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['erfSize'], y = train['price'])

plt.ylabel('price', fontsize=13)

plt.xlabel('erfSize', fontsize=13)

plt.show()
# Dropping variables seen as outliers from the data 

train = train.drop(train[(train['erfSize']>8000) & (train['price']<3000000)].index)



fig, ax = plt.subplots()

ax.scatter(train['erfSize'], train['price'])

plt.ylabel('price', fontsize=13)

plt.xlabel('erfSize', fontsize=13)

plt.show()
# Concatinating our train and test sets to avoid a mismatch when fitting the model

ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train['price'].values

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop('price', axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
# Checking percentage of missing data

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head(20)
# visualising missing values

sns.set_style("white")

f, ax = plt.subplots(figsize=(8, 7))

sns.set_color_codes(palette='deep')

missing = round(train.isnull().mean()*100,2)

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar(color="b")

# Plotting the visual presentation

ax.xaxis.grid(False)

ax.set(ylabel="Percent of missing values")

ax.set(xlabel="Features")

ax.set(title="Percent missing data by feature")

sns.despine(trim=True, left=True)
# Correlation heatmap to see how variables are correlated with our target variable

corrmat = train.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True, annot=True)
#Drop columns 'bedroom', 'bathroom'

train.drop(['bedroom', 'bathroom'],axis =1, inplace = True)

test.drop(['bedroom', 'bathroom'],axis =1, inplace = True)
# mark zero values as missing or NaN

all_data[['buildingSize','erfSize']] = all_data[['buildingSize','erfSize']].replace(0, np.NaN)

# fill missing values with mean column values

all_data.fillna(all_data.mean(), inplace=True)
# Checking if there are any missing values after replacing them with the mean

all_data[['buildingSize','bathroom','bedroom','erfSize']].isnull().sum()
# Replacing missing values in 'garage' with "None"

all_data["garage"] = all_data["garage"].fillna("None")
# Checking if there are any missing values left

print(all_data.isnull().sum())
#Label Encoding categorical variables that may contain information in their ordering set.

cat_cols = ('area','data-isonshow','data-location','type')



# process columns, apply LabelEncoder to categorical features

for c in cat_cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))



# shape        

print('Shape all_data: {}'.format(all_data.shape))
# Finding all numeric variables in the data

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics2 = []

for i in all_data.columns:

    if all_data[i].dtype in numeric_dtypes:

        numerics2.append(i)
# Checking the skewness of our numerical variables

sns.set_style("white")

f, ax = plt.subplots(figsize=(8, 7))

ax.set_xscale("log")

ax = sns.boxplot(data=all_data[numerics2] , orient="h", palette="Set1")



# Plotting the visual presentation

ax.xaxis.grid(False)

ax.set(ylabel="Feature names")

ax.set(xlabel="Numeric values")

ax.set(title="Numeric Distribution of Features")

sns.despine(trim=True, left=True)
# Find the skewed  numerical features

skew_features = all_data[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)



high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index

print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))

skewness = pd.DataFrame({'Skew' :high_skew})

skew_features.head(10)
# Normalise skewed features

for i in skew_index:

    all_data[i] = boxcox1p(all_data[i], boxcox_normmax(all_data[i] + 1))
sns.set_style("white")

f, ax = plt.subplots(figsize=(8, 7))

ax.set_xscale("log")

ax = sns.boxplot(data=all_data[skew_index] , orient="h", palette="Set1")

# Tweak the visual presentation

ax.xaxis.grid(False)

ax.set(ylabel="Feature names")

ax.set(xlabel="Numeric values")

ax.set(title="Numeric Distribution of Features")

sns.despine(trim=True, left=True)
final_features = pd.get_dummies(all_data).reset_index(drop=True)

print('Features size:', all_data.shape)

final_features.head()
# Getting the new train and test sets by splliting the concatenated data.

train = all_data[:ntrain]

test = all_data[ntrain:]
X = train

y = y_train
X.shape, test.shape
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

test_scaled = scaler.fit_transform(test)

X_scaled.shape, test_scaled.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, 

                                                    y, 

                                                    test_size=0.20, 

                                                    shuffle=False)
X_train.shape
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.01)
lasso.fit(X_train,y_train)
intercept = float(lasso.intercept_)
coeff = pd.DataFrame(lasso.coef_, X.columns, columns=['Coefficient'])
print("Intercept:", float(intercept))
coeff
from sklearn import metrics
predictedTrainPrices = lasso.predict(X_train)

trainR2 = metrics.r2_score(y_train, predictedTrainPrices)

predictedTestPrices = lasso.predict(X_test)

testR2 = metrics.r2_score(y_test, predictedTestPrices)



print("Trained R-squared: ",trainR2)

print("Test R-squared: ",testR2)
test_lasso = lasso.predict(test_scaled)
test_lasso
predict = np.exp(test_lasso)
predict
train_ID.shape
#model = pd.DataFrame()

#model['house-id'] = house_ID

#model['price'] = predict

#model.to_csv('..\predicted_submission.csv',index=False)
#model