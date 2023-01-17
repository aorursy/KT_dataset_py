# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split



# Model

from sklearn.linear_model import LinearRegression



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# read in all data

train = pd.read_csv('/kaggle/input/dsn-ai-futa-challenge/Train.csv')

test = pd.read_csv('/kaggle/input/dsn-ai-futa-challenge/Test.csv')

submission = pd.read_csv('/kaggle/input/dsn-ai-futa-challenge/Sample_Submission.csv')
#print out the shape of both Train and Test data

print('Train shape',train.shape)

print('Test shape',test.shape)
#Print the head of train

train.head()
#Print the head of test

test.head()
#Total number of entries that are missing in each column

train.isna().sum()
plt.figure(figsize=(10,6)) # Set the size of the plot

sns.heatmap(train.corr(),annot=True) # Correlation Heatmap
#scatterplot of all features

cat_col = ['Product_Fat_Content','Product_Type','Supermarket_Location_Type','Supermarket_Type']#get categorical features of train data



for columns in cat_col: 

    sns.set()

    cols = ['Product_Identifier', 'Supermarket_Identifier',

           'Product_Fat_Content', 'Product_Shelf_Visibility', 'Product_Type',

           'Product_Price', 'Supermarket_Opening_Year',

           'Supermarket_Location_Type', 'Supermarket_Type',

           'Product_Supermarket_Sales']

    plt.figure()

    sns.pairplot(train[cols], size = 3.0, hue=columns)

    plt.show()

# the columns contain missing value are 1.Product_Weight(Numerical) 2. Supermarket _Size (Categorical)

train['Product_Weight'].fillna(train['Product_Weight'].mean(),inplace=True)

train['Supermarket _Size'].fillna(train['Supermarket _Size'].mode()[0],inplace=True)



# We will have to use the same strategy for out test data

test['Product_Weight'].fillna(test['Product_Weight'].mean(),inplace=True)

test['Supermarket _Size'].fillna(test['Supermarket _Size'].mode()[0],inplace=True)
#Now we have no missing values in both train and test data

train.isna().sum()
test.isna().sum()
# Concatenate train and test sets for easy feature engineering.

# You can as well apply the transformations separately on the train and test data intead of concatenating them.

ntrain = train.shape[0]

ntest = test.shape[0]



#get target variable

y = train['Product_Supermarket_Sales']



all_data = pd.concat((train,test)).reset_index(drop=True)



#drop target variable

all_data.drop(['Product_Supermarket_Sales'], axis=1, inplace=True)



print("Total data size is : {}".format(all_data.shape))
# Let's Create the squarred root of Product_Price

all_data['Product_Price_sqrt'] = np.sqrt(all_data['Product_Price'])



#Create some cross features

all_data['cross_Price_weight'] = all_data['Product_Price'] * all_data['Product_Weight']
all_data.columns
one_hot_cols = ['Supermarket_Type','Supermarket _Size','Product_Type','Supermarket_Location_Type']



label_cols = ['Product_Identifier','Supermarket_Identifier','Product_Fat_Content']
all_data = pd.get_dummies(all_data,prefix_sep="_",columns=one_hot_cols)
for col in label_cols:

    all_data[col] = all_data[col].factorize()[0]
# We are going to drop Product_Supermarket_Identifier' since it's just an ID and we don't need it.

all_data.drop('Product_Supermarket_Identifier',axis=1,inplace=True)
#Lets get the new train and test set

train = all_data[:ntrain]

test = all_data[ntrain:]



print('Train size: ' + str(train.shape))

print('Test size: ' + str(test.shape))
train.head()
test.head()
# Spllitting train data into training and validation set. We are using just 20%(0.2) for validation 

X_train,X_test,y_train,y_test = train_test_split(train,y,test_size=0.2,random_state=42)
# Define the model

lr = LinearRegression()
lr.fit(X_train,y_train)
y_hat = lr.predict(X_test)
print('Validation scores', np.sqrt(mean_squared_error(y_test, y_hat)))



print('Training scores', np.sqrt(mean_squared_error(y_train, lr.predict(X_train))))
test_pred = lr.predict(test);test_pred
submission.head()
submission['Product_Supermarket_Sales'] = test_pred
submission.to_csv('first_submission.csv',index=False)

#If you submit this you should at least find a better position on the LeaderBoard