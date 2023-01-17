# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/pandas-bokeh/dataset_tk.csv')

df.head()
# Numerical features

Numerical_feat = [feature for feature in df.columns if df[feature].dtypes != 'O']

print('Total numerical features: ', len(Numerical_feat))

print('\nNumerical Features: ', Numerical_feat)
# Zoomed heatmap, correlation matrix

sns.set(rc={'figure.figsize':(12,8)})

correlation_matrix = df.corr()



k = 10             #number of variables for heatmap

cols = correlation_matrix.nlargest(k, 'Rajasthan')['Rajasthan'].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
## Lets Find the realtionship between discrete features and SalePrice



#plt.figure(figsize=(8,6))



for feature in Numerical_feat:

    data=df.copy()

    plt.figure(figsize=(8,6))

    data.groupby(feature)['Rajasthan'].median().plot.bar()

    plt.xlabel(feature)

    plt.ylabel('Rajasthan')

    plt.title(feature)

    plt.show()
df[Numerical_feat].hist(bins=25)

plt.show()
## let us now examine the relationship between continuous features and SalePrice

## Before that lets find continous features that donot contain zero values



continuous_nozero = [feature for feature in Numerical_feat if 0 not in data[feature].unique() and feature not in ['Tripura', 'Nagaland']]



for feature in continuous_nozero:

    plt.figure(figsize=(8,6))

    data = df.copy()

    data[feature] = np.log(data[feature])

    data['Rajasthan'] = np.log(data['Rajasthan'])

    plt.scatter(data[feature], data['Rajasthan'])

    plt.xlabel(feature)

    plt.ylabel('Rajasthan')

    plt.show()
## Normality and distribution checking for continous features

for feature in continuous_nozero:

    plt.figure(figsize=(6,6))

    data = df.copy()

    sns.distplot(data[feature])

    plt.show()
# categorical features

categorical_feat = [feature for feature in df.columns if df[feature].dtypes=='O']

print('Total categorical features: ', len(categorical_feat))

print('\n',categorical_feat)
# lets find unique values in each categorical features

for feature in categorical_feat:

    print('{} has {} categories. They are:'.format(feature,len(df[feature].unique())))

    print(df[feature].unique())

    print('\n')
# let us find relationship of categorical with target variable



for feature in categorical_feat:

    data=df.copy()

    data.groupby(feature)['Rajasthan'].median().plot.bar()

    plt.xlabel(feature)

    plt.ylabel('Rajasthan')

    plt.title(feature)

    plt.show()
# these are selected features from EDA section

features = ['Rajasthan', 'Punjab', 'Uttarakhand', 'UP', 'J&K', 'Meghalaya', 'DNH', 'Unnamed: 0']
# plot bivariate distribution (above given features with saleprice(target feature))

for feature in features:

    if feature!='Rajasthan':

        plt.scatter(df[feature], df['Rajasthan'])

        plt.xlabel(feature)

        plt.ylabel('Rajasthan')

        plt.show()
#Deleting outliers for GrLivArea

df = df.drop(df[(df['Uttarakhand']>4000) & (df['Rajasthan']<300000)].index)



plt.scatter(df['Uttarakhand'], df['Rajasthan'])

plt.xlabel('Uttarakhand')

plt.ylabel('Rajasthan')

plt.show()
# these are selected features from EDA section

features = ['Rajasthan', 'Punjab', 'Uttarakhand', 'UP', 'J&K', 'Meghalaya', 'DNH']



# selecting continuous features from above

continuous_features = ['Rajasthan', 'Uttarakhand']
#Train = train_df.shape[0]

#Test = test_df.shape[0]

#target_feature = train_df.SalePrice.values

#combined_data = pd.concat((train_df, test_df)).reset_index(drop=True)

#combined_data.drop(['SalePrice','Id'], axis=1, inplace=True)

#print("all_data size is : {}".format(combined_data.shape))
#Since I have no train, test files, Id, I adapted the code above for just 1 line, so that I could plot the distplot.  

combined_data = pd.concat((df, df)).reset_index(drop=True)
from scipy.stats import norm



# checking distribution of continuous features(histogram plot)

for feature in continuous_features:

    if feature!='Rajasthan':

        sns.distplot(combined_data[feature], fit=norm)

        plt.show()

    else:

        sns.distplot(df['Rajasthan'], fit=norm)

        plt.show()
# so let's label encode above ordinal features

from sklearn.preprocessing import LabelEncoder

for feature in features:

    encoder = LabelEncoder()

    combined_data[feature] = encoder.fit_transform(list(combined_data[feature].values))
# Now lets see label encoded data

combined_data[features].head()
## One hot encoding or getting dummies 



dummy_ordinals = pd.get_dummies(features) 

dummy_ordinals.head()
# creating dummy variables



combined_data = pd.get_dummies(combined_data)

print(combined_data.shape)
combined_data.head()
# let's first see descriptive stat info 

combined_data.describe()
## we willtake all features from combined_dummy_data 

features_to_scale = [feature for feature in combined_data]

print(len(features_to_scale))
## Now here is where we will scale our data using sklearn module.



from sklearn.preprocessing import MinMaxScaler



cols = combined_data.columns  # columns of combined_dummy_data



scaler = MinMaxScaler()

combined_data = scaler.fit_transform(combined_data[features_to_scale])
# after scaling combined_data it is now in ndarray datypes

# so we will create DataFrame from it

combined_scaled_data = pd.DataFrame(combined_data, columns=[cols])
combined_scaled_data.head() # this is the same combined_dummy_data in scaled form.
# lets see descriptive stat info 

combined_scaled_data.describe()
#That's the code. Though we don't have train nor test, then I adapted once more. 

#train_df.shape, test_df.shape, combined_scaled_data.shape, combined_data.shape
df.shape, df.shape, combined_scaled_data.shape, combined_data.shape
# separate train data and test data 

train_data = combined_scaled_data.iloc[:504,:]

test_data = combined_scaled_data.iloc[504:,:]



train_data.shape, test_data.shape
## lets add target feature to train_data

#train_data['Rajasthan']= train_data['Rajasthan']  # This saleprice is normalized. Its very impportant
train_data = train_data

train_data.head(10)
test_data = test_data.reset_index()

test_data.tail()
dataset = train_data.copy()  # copy train_data to dataset variable
dataset.head()
dataset = dataset.dropna()
## lets create dependent and target feature vectors



X = dataset.drop(['Rajasthan'],axis=1)

Y = dataset[['Rajasthan']]



X.shape, Y.shape
Y.head()
# lets do feature selection here



from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression



# define feature selection

fs = SelectKBest(score_func=f_regression, k=27)

# apply feature selection

X_selected = fs.fit_transform(X, Y)

print(X_selected.shape)
cols = list(range(1,28))



## create dataframe of selected features



selected_feat = pd.DataFrame(data=X_selected,columns=[cols])

selected_feat.head()
# perform train_test_split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(selected_feat,Y,test_size=0.3,random_state=0)
x_train.shape, x_test.shape
from sklearn.linear_model import LinearRegression

from sklearn import metrics



lr = LinearRegression()

lr.fit(x_train,y_train)
y_pred = lr.predict(x_test) # predicting test data

y_pred[:10]
# Evaluating the model

print('R squared score',metrics.r2_score(y_test,y_pred))



print('\nMean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# check for underfitting and overfitting

print('Train Score: ', lr.score(x_train,y_train))

print('Test Score: ', lr.score(x_test,y_test))
## scatter plot of original and predicted target test data

plt.figure(figsize=(8,6))

plt.scatter(y_test,y_pred)

plt.xlabel('y_tes')

plt.ylabel('y_pred')

plt.show()
## Lets do error plot

## to get error in prediction just substract predicted values from original values



error = list(y_test.values-y_pred)

plt.figure(figsize=(8,6))

sns.distplot(error)
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=100)

rf_reg.fit(x_train,y_train)
y_pred = rf_reg.predict(x_test)

print(y_pred[:10])
## evaluating the model



print('R squared error',metrics.r2_score(y_test,y_pred))



print('\nMean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# check score

print('Train Score: ', rf_reg.score(x_train,y_train))

print('Test Score: ', rf_reg.score(x_test,y_test))
## scatter plot of original and predicted target test data

plt.figure(figsize=(8,6))

plt.scatter(y_test,y_pred)

plt.xlabel('y_tes')

plt.ylabel('y_pred')

plt.show()
## Lets do error plot

## to get error in prediction just substract predicted values from original values



error = list(y_test.values-y_pred)

plt.figure(figsize=(8,6))

sns.distplot(error)
# Plot

sns.set_style('whitegrid')

fig, (ax1,ax2) = plt.subplots(1, 2)

fig.set_size_inches(18, 6)



sns.countplot(df['Punjab'], order=df['Punjab'].value_counts().index[:20],palette='viridis', ax=ax1)

sns.countplot(df['Rajasthan'], order=df['Rajasthan'].value_counts().index[:20],palette='viridis', ax=ax2)



ax1.tick_params(axis='x', labelrotation=45)

ax2.tick_params(axis='x', labelrotation=45)

ax1.set_title('Punjab')

ax2.set_title('Tripura')

ax2.set(ylim=(0, 100))





plt.show()