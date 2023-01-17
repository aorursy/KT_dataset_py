# imports
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Using UCI repository data
df = pd.read_csv('../input/trains-transformed.data',delim_whitespace=True,header=None)
df.head()
df.shape
print(df.columns)
# count of target variables
print(df.iloc[:,32].value_counts())
# find nulls in the data
df.isnull().sum().sum()
# find NA in the data
df.isna().sum().sum()
print('The sum of all the ? symbols in column 3 is',df.iloc[:,3].str.contains('\-').sum())
print('The sum of all the ? symbols in column 4 is',df.iloc[:,4].str.contains('\-').sum())
print('The sum of all the ? symbols in column 6 is',df.iloc[:,6].str.contains('\-').sum())
print('The sum of all the ? symbols in column 8 is',df.iloc[:,8].str.contains('\-').sum())
print('The sum of all the ? symbols in column 9 is',df.iloc[:,9].str.contains('\-').sum())
print('The sum of all the ? symbols in column 11 is',df.iloc[:,11].str.contains('\-').sum())
print('The sum of all the ? symbols in column 12 is',df.iloc[:,12].str.contains('\-').sum())
print('The sum of all the ? symbols in column 13 is',df.iloc[:,13].str.contains('\-').sum())
print('The sum of all the ? symbols in column 14 is',df.iloc[:,14].str.contains('\-').sum())
print('The sum of all the ? symbols in column 15 is',df.iloc[:,15].str.contains('\-').sum())
print('The sum of all the ? symbols in column 16 is',df.iloc[:,16].str.contains('\-').sum())
print('The sum of all the ? symbols in column 17 is',df.iloc[:,17].str.contains('\-').sum())
print('The sum of all the ? symbols in column 18 is',df.iloc[:,18].str.contains('\-').sum())
print('The sum of all the ? symbols in column 19 is',df.iloc[:,19].str.contains('\-').sum())
print('The sum of all the ? symbols in column 20 is',df.iloc[:,20].str.contains('\-').sum())
print('The sum of all the ? symbols in column 21 is',df.iloc[:,21].str.contains('\-').sum())
# Replacing all the '-' with NaN.
df.iloc[:,12].replace('-',np.nan,inplace=True)
df.iloc[:,13].replace('-',np.nan,inplace=True)
df.iloc[:,14].replace('-',np.nan,inplace=True)
df.iloc[:,15].replace('-',np.nan,inplace=True)
df.iloc[:,16].replace('-',np.nan,inplace=True)
df.iloc[:,17].replace('-',np.nan,inplace=True)
df.iloc[:,18].replace('-',np.nan,inplace=True)
df.iloc[:,19].replace('-',np.nan,inplace=True)
df.iloc[:,20].replace('-',np.nan,inplace=True)
df.iloc[:,21].replace('-',np.nan,inplace=True)
# checking if all the '-' have been removed.
print('The sum of all the ? symbols in column 3 is',df.iloc[:,3].str.contains('\-').sum())
print('The sum of all the ? symbols in column 4 is',df.iloc[:,4].str.contains('\-').sum())
print('The sum of all the ? symbols in column 6 is',df.iloc[:,6].str.contains('\-').sum())
print('The sum of all the ? symbols in column 8 is',df.iloc[:,8].str.contains('\-').sum())
print('The sum of all the ? symbols in column 9 is',df.iloc[:,9].str.contains('\-').sum())
print('The sum of all the ? symbols in column 11 is',df.iloc[:,11].str.contains('\-').sum())
print('The sum of all the ? symbols in column 12 is',df.iloc[:,12].str.contains('\-').sum())
print('The sum of all the ? symbols in column 13 is',df.iloc[:,13].str.contains('\-').sum())
print('The sum of all the ? symbols in column 14 is',df.iloc[:,14].str.contains('\-').sum())
print('The sum of all the ? symbols in column 15 is',df.iloc[:,15].str.contains('\-').sum())
print('The sum of all the ? symbols in column 16 is',df.iloc[:,16].str.contains('\-').sum())
print('The sum of all the ? symbols in column 17 is',df.iloc[:,17].str.contains('\-').sum())
print('The sum of all the ? symbols in column 18 is',df.iloc[:,18].str.contains('\-').sum())
print('The sum of all the ? symbols in column 19 is',df.iloc[:,19].str.contains('\-').sum())
print('The sum of all the ? symbols in column 20 is',df.iloc[:,20].str.contains('\-').sum())
print('The sum of all the ? symbols in column 21 is',df.iloc[:,21].str.contains('\-').sum())
# In the entire dataset we have 51 NaN values(Missing values)
print('count of all the missing values in the data is',df.isna().sum().sum())
#Replacing all the NaN with forward values.
df.fillna(method='ffill',inplace=True) 
#Checking if the values are replacced.
print('count of all the missing values after treating the data is',df.isna().sum().sum())
# Label encoding all the strings
enc_3 = LabelEncoder()
df.iloc[:,3] = enc_3.fit_transform(df.iloc[:,3])
enc_4 = LabelEncoder()
df.iloc[:,4] = enc_4.fit_transform(df.iloc[:,4])
enc_6 = LabelEncoder()
df.iloc[:,6] = enc_6.fit_transform(df.iloc[:,6])
enc_8 = LabelEncoder()
df.iloc[:,8] = enc_8.fit_transform(df.iloc[:,8])
enc_9 = LabelEncoder()
df.iloc[:,9] = enc_9.fit_transform(df.iloc[:,9])
enc_11 = LabelEncoder()
df.iloc[:,11] = enc_11.fit_transform(df.iloc[:,11])
enc_12 = LabelEncoder()
df.iloc[:,12] = enc_12.fit_transform(df.iloc[:,12])
enc_13 = LabelEncoder()
df.iloc[:,13] = enc_13.fit_transform(df.iloc[:,13])
enc_14 = LabelEncoder()
df.iloc[:,14] = enc_14.fit_transform(df.iloc[:,14])
enc_15 = LabelEncoder()
df.iloc[:,15] = enc_15.fit_transform(df.iloc[:,15])
enc_16 = LabelEncoder()
df.iloc[:,16] = enc_16.fit_transform(df.iloc[:,16])
enc_17 = LabelEncoder()
df.iloc[:,17] = enc_17.fit_transform(df.iloc[:,17])
enc_18 = LabelEncoder()
df.iloc[:,18] = enc_18.fit_transform(df.iloc[:,18])
enc_20 = LabelEncoder()
df.iloc[:,20] = enc_20.fit_transform(df.iloc[:,20])
enc_19 = LabelEncoder()
df.iloc[:,19] = enc_19.fit_transform(df.iloc[:,19])
enc_21 = LabelEncoder()
df.iloc[:,21] = enc_21.fit_transform(df.iloc[:,21])
df.info()
X,y = df.iloc[:,0:32],df.iloc[:,32]
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Before applying PCA the variables must be scaled
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Applying PCA
pca_none = PCA(n_components = None)
X_train_none = pca_none.fit_transform(X_train)
X_test_none = pca_none.transform(X_test)
explained_variance = pca_none.explained_variance_ratio_
print(explained_variance*100)
print((explained_variance*100).round(2))
# Applying PCA to convert the features into 4 dimensions only.
pca = PCA(n_components = 4)
X_train_4 = pca.fit_transform(X_train)
X_test_4 = pca.transform(X_test)
#explained_variance = pca.explained_variance_ratio_
# We can also use Yellowbrick API for PCA
from yellowbrick.features.pca import PCADecomposition
visualizer = PCADecomposition()
visualizer.fit_transform(X_train)
#Using poof command we can easily visualize the PCA plot
visualizer.poof()
