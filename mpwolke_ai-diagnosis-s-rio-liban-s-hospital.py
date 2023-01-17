# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
df = pd.read_excel('/kaggle/input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')

df.head()
#Correlation map to see how features are correlated with each other and with SalePrice

corrmat = df.corr(method='kendall')

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
# Numerical features

Numerical_feat = [feature for feature in df.columns if df[feature].dtypes != 'O']

print('Total numerical features: ', len(Numerical_feat))

print('\nNumerical Features: ', Numerical_feat)
# Let's find the null values in data



total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
## Lets Find the realtionship between discrete features and ALBUMIN_MAX



#plt.figure(figsize=(8,6))



for feature in Numerical_feat:

    data=df.copy()

    plt.figure(figsize=(8,6))

    data.groupby(feature)['HEMATOCRITE_MAX'].median().plot.bar()

    plt.xlabel(feature)

    plt.ylabel('HEMATOCRITE_MAX')

    plt.title(feature)

    plt.show()
#df[Numerical_feat].hist(bins=25)

#plt.show()
## let us now examine the relationship between continuous features and SalePrice

## Before that lets find continous features that donot contain zero values



continuous_nozero = [feature for feature in Numerical_feat if 0 not in data[feature].unique() and feature not in ['BE_VENOUS_DIFF', 'BIC_ARTERIAL_DIFF']]



for feature in continuous_nozero:

    plt.figure(figsize=(8,6))

    data = df.copy()

    data[feature] = np.log(data[feature])

    data['HEMATOCRITE_MAX'] = np.log(data['HEMATOCRITE_MAX'])

    plt.scatter(data[feature], data['HEMATOCRITE_MAX'])

    plt.xlabel(feature)

    plt.ylabel('HEMATOCRITE_MAX')

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

    data.groupby(feature)['HEMATOCRITE_MAX'].median().plot.bar()

    plt.xlabel(feature)

    plt.ylabel('')

    plt.title(feature)

    plt.show()
# these are selected features from EDA section

features = ['HEMATOCRITE_MAX', 'LEUKOCYTES_MAX', 'HEMOGLOBIN_MAX', 'NEUTROPHILES_MAX', 'TGO_MAX', 'TGP_MAX', 'RESPIRATORY_RATE_DIFF_REL', 'TEMPERATURE_DIFF_REL']
# plot bivariate distribution (above given features with saleprice(target feature))

for feature in features:

    if feature!='HEMATOCRITE_MAX':

        plt.scatter(df[feature], df['HEMATOCRITE_MAX'])

        plt.xlabel(feature)

        plt.ylabel('HEMATOCRITE_MAX')

        plt.show()
# Lets first handle numerical features with nan value

numerical_nan = [feature for feature in df.columns if df[feature].isna().sum()>1 and df[feature].dtypes!='O']

numerical_nan
df[numerical_nan].isna().sum()
## Replacing the numerical Missing Values



for feature in numerical_nan:

    ## We will replace by using median since there are outliers

    median_value=df[feature].median()

    

    df[feature].fillna(median_value,inplace=True)

    

df[numerical_nan].isnull().sum()
# categorical features with missing values

categorical_nan = [feature for feature in df.columns if df[feature].isna().sum()>1 and df[feature].dtypes=='O']

print(categorical_nan)
df[categorical_nan].isna().sum()
# replacing missing values in categorical features

for feature in categorical_nan:

    df[feature] = df[feature].fillna('None')
df[categorical_nan].isna().sum()
#Deleting outliers for LEUKOCYTES_MAX

df = df.drop(df[(df['LEUKOCYTES_MAX']>4000) & (df['HEMATOCRITE_MAX']<300000)].index)



plt.scatter(df['LEUKOCYTES_MAX'], df['HEMATOCRITE_MAX'])

plt.xlabel('LEUKOCYTES_MAX')

plt.ylabel('HEMATOCRITE_MAX')

plt.show()
#Deleting outliers for HEMOGLOBIN_MAX

df = df.drop(df[(df['HEMOGLOBIN_MAX']>4000) & (df['HEMATOCRITE_MAX']<300000)].index)



plt.scatter(df['HEMOGLOBIN_MAX'], df['HEMATOCRITE_MAX'])

plt.xlabel('HEMOGLOBIN_MAX')

plt.ylabel('HEMATOCRITE_MAX')

plt.show()
#Deleting outliers for NEUTROPHILES_MAX 

df = df.drop(df[(df['NEUTROPHILES_MAX']>4000) & (df['HEMATOCRITE_MAX']<300000)].index)



plt.scatter(df['NEUTROPHILES_MAX'], df['HEMATOCRITE_MAX'])

plt.xlabel('NEUTROPHILES_MAX')

plt.ylabel('HEMATOCRITE_MAX')

plt.show()
#Deleting outliers for TGO_MAX

df = df.drop(df[(df['TGO_MAX']>4000) & (df['HEMATOCRITE_MAX']<300000)].index)



plt.scatter(df['TGO_MAX'], df['HEMATOCRITE_MAX'])

plt.xlabel('TGO_MAX')

plt.ylabel('HEMATOCRITE_MAX')

plt.show()
#Deleting outliers for TGP_MAX 

df = df.drop(df[(df['TGP_MAX']>4000) & (df['HEMATOCRITE_MAX']<300000)].index)



plt.scatter(df['TGP_MAX'], df['HEMATOCRITE_MAX'])

plt.xlabel('TGP_MAX')

plt.ylabel('HEMATOCRITE_MAX')

plt.show()
#Deleting outliers for RESPIRATORY_RATE_DIFF_REL 

df = df.drop(df[(df['RESPIRATORY_RATE_DIFF_REL']>4000) & (df['HEMATOCRITE_MAX']<300000)].index)



plt.scatter(df['RESPIRATORY_RATE_DIFF_REL'], df['HEMATOCRITE_MAX'])

plt.xlabel('RESPIRATORY_RATE_DIFF_REL')

plt.ylabel('HEMATOCRITE_MAX')

plt.show()
#Deleting outliers for TEMPERATURE_DIFF_REL

df = df.drop(df[(df['TEMPERATURE_DIFF_REL']>4000) & (df['HEMATOCRITE_MAX']<300000)].index)



plt.scatter(df['TEMPERATURE_DIFF_REL'], df['HEMATOCRITE_MAX'])

plt.xlabel('TEMPERATURE_DIFF_REL')

plt.ylabel('HEMATOCRITE_MAX')

plt.show()
# these are selected features from EDA section

features = ['HEMATOCRITE_MAX', 'LEUKOCYTES_MAX', 'HEMOGLOBIN_MAX', 'NEUTROPHILES_MAX', 'TGO_MAX', 'TGP_MAX', 'RESPIRATORY_RATE_DIFF_REL', 'TEMPERATURE_DIFF_REL']



# selecting continuous features from above

continuous_features = ['HEMATOCRITE_MAX', 'LEUKOCYTES_MAX', 'HEMOGLOBIN_MAX', 'NEUTROPHILES_MAX', 'TGO_MAX', 'TGP_MAX', 'RESPIRATORY_RATE_DIFF_REL', 'TEMPERATURE_DIFF_REL']
#Train = train_df.shape[0]

#Test = test_df.shape[0]

#target_feature = train_df.SalePrice.values

#combined_data = pd.concat((train_df, test_df)).reset_index(drop=True)

#combined_data.drop(['SalePrice','Id'], axis=1, inplace=True)

#print("all_data size is : {}".format(combined_data.shape))
#Since I have no train, test files, Id, I adapted the code above for just 1 line, so that I could plot the distplot.  

combined_data = pd.concat((df, df)).reset_index(drop=True)
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
# checking distribution of continuous features(histogram plot)

for feature in continuous_features:

    if feature!='HEMATOCRITE_MAX':

        sns.distplot(combined_data[feature], fit=norm)

        plt.show()

    else:

        sns.distplot(df['HEMATOCRITE_MAX'], fit=norm)

        plt.show()
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

#train_data['deaths_per_million_85_days_after_first_death']= train_data['deaths_per_million_85_days_after_first_death']  # This saleprice is normalized. Its very impportant
train_data = train_data

train_data.head(10)
test_data = test_data.reset_index()

test_data.tail()
dataset = train_data.copy()  # copy train_data to dataset variable
dataset.head()
dataset = dataset.dropna()
## lets create dependent and target feature vectors



X = dataset.drop(['HEMATOCRITE_MAX'],axis=1)

Y = dataset[['HEMATOCRITE_MAX']]



X.shape, Y.shape
Y.head()