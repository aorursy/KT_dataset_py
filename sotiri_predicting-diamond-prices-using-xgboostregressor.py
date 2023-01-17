# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

import xgboost as xgb

from category_encoders.ordinal import OrdinalEncoder

from sklearn.metrics import r2_score

from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

from sklearn_pandas import DataFrameMapper

import numpy as np
#Get data:

file_path = '../input/diamonds.csv'



df = pd.read_csv(file_path,index_col =0)

# Check for completness of dataset: No null entires!

df.info()
# Print .head() to get a flavour of the dataset:

df.head()
# Count the types of cut: Ideal and premium cuts are most popular

sns.countplot(df.cut)
# In order to do any feature engineering before building the model

# Its usually a good idea to see which attributes strongly correlated with Price

# Correlation:

df_numeric = df.select_dtypes(exclude=['object'])

df_corr = df_numeric.corr()

plt.figure(figsize=(10,10))

#Heatmap:

sns.set(font_scale=1.5)

sns.heatmap(df_corr,center=0,cmap='YlGnBu',square=True, annot=True)



#Carat is correlated the most, although x,y,z are not far behind.
sns.pairplot(df)
# Note: We want to maybe create new features such as ratio of carat to length

# Ratio of length to width

# See if any of the above also correlate with price

# But before we can do that you may notice x,y,z suffer from zero values

# We don't want to divide by zero!

sns.jointplot(x='carat',y='price',data=df)

sns.jointplot(x='x',y='price',data=df)

sns.jointplot(x='x',y='y',data=df)
# We have value where length=0,width=0,depth=0. Remove this before feature engineering otherwise we divide by 0 and get inf:

print((df.x == 0).any())

print((df.y == 0).any())

print((df.z == 0).any())
#Delete each row with this condition:

df = df[df['x']!= 0]

df = df[df['y']!= 0]

df = df[df['z']!= 0]
# Feature engineering: Length to width ratio, carat/x,carat,y,carat/z

df = df.assign(length_width_ratio = round(df.x/df.y,2))

df = df.assign(carat_length_ratio = round(df.carat/df.x,2))

df = df.assign(carat_width_ratio = round(df.carat/df.y,2))

df = df.assign(carat_depth_ratio = round(df.carat/df.z,2))
# Now check updated correlations with price:

# length to width ratio has little correlation but the resut are very highly correlated!

df_corr = df.corr()

df_corr['price']
# Before we go anyfurther into preprocessing its good practice to reserve a holdout set for testing

# We will return to this later!

diamond_train,diamond_test = train_test_split(df,test_size=0.2)
# Good idea now is to only work with our training data and start the preprocessing stage

# X=features, y=target

X,y = diamond_train.drop('price',axis=1),diamond_train.price

# This section we will attempt to use OrdinalEncoder() which can be customerised so that I can specify what each class in each attribute should be labeled as.

# We will also standardize our numerical attributes before inputing into our model (standard ML practice).



# List of dictionaries, first key should be the feature name, second key should be mapping.



ordinal_enc_mapping = [{'col':'cut','mapping': [('Fair',0),('Good',1),('Very Good',2),('Premium',3),('Ideal',4)]},

                       {'col':'color','mapping': [('J',0),('I',1),('H',2),('G',3),('F',4),('E',5),('D',6)]},

                       {'col':'clarity','mapping': [('I1',0),('SI2',1),('SI1',2),('VS2',3),('VS1',4),('VVS2',5),('VVS1',6),('IF',7)]}]



# Seperate categorical columns from numerical:

categorical_columns = list(X.select_dtypes(include=['object']).columns)

numerical_columns = list(X.select_dtypes(exclude=['object']).columns)



#Dataframemapper: Useful for applying transformation to specific columns of the dataframe.

#numeric_mapper will apply transforms to the numeric columns selected above. First element in tuple is a list of columns to apply transformation to and second element is transformation.

numeric_mapper = DataFrameMapper([([numeric_feature], StandardScaler()) for numeric_feature in numerical_columns],

                                sparse=False, df_out=True,input_df=True)





#Apply feature union to both dataframe mapper and OrdinalEncoder:



numerical_categorical_union = FeatureUnion([('num_mapper',numeric_mapper),('oe',OrdinalEncoder(mapping=ordinal_enc_mapping,cols=categorical_columns))])
# Now we setup a Pipeline to run this badboy in! 

# Piece together into a pipeline: I have set the n_estimators high, rarely do we have an issue with overfitting on XGBoost if we keep the learning rate at a sufficient level.

steps = [('featureunion',numerical_categorical_union),

         ('xgb_model',xgb.XGBRegressor(n_estimators=1000,subsample=0.3,max_depth=10,learning_rate=0.005,gamma=1.45))]



pipeline = Pipeline(steps)
#cross validate: 3 folds

cv = cross_val_score(pipeline,X,y,cv=3,verbose=1,scoring='r2')
#R2 score: Average of 3 folds

print(cv.mean())
# Now to the above performs well on the training data, lets check the test data with 3 folds:

X_test,y_test = diamond_test.drop('price',axis=1),diamond_test.price



cv_test = cross_val_score(pipeline,X_test,y_test,scoring='r2',cv=3)



#R2 score:

print('R2 score:',cv_test.mean())