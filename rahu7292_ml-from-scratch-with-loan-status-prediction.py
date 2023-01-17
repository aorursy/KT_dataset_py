# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import init_notebook_mode, download_plotlyjs, iplot

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()

import warnings

warnings.filterwarnings('ignore')





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print()

print("The files in the dataset are:-")

from subprocess import check_output

print(check_output(['ls','../input']).decode('utf'))



# Any results you write to the current directory are saved as output.
# Importing the dataset.

df_train = pd.read_csv("../input/credit_train.csv")

df_test = pd.read_csv("../input/credit_test.csv")
# Let us check the top 5 entries in training dataset.

df_train.head()
# We need to convert the values of years in current job into integer format.

df_train['Years in current job'] = df_train['Years in current job'].map({'8 years':8, '10+ years':15,

                                        '3 years':3, '5 years':5, '< 1 year':0.5, 

                            '2 years':2, '4 years':4, '9 years':9, '7 years':7, '1 year':1, '6 years':6})



df_test['Years in current job'] = df_test['Years in current job'].map({'8 years':8, '10+ years':15,

                                        '3 years':3, '5 years':5, '< 1 year':0.5, 

                            '2 years':2, '4 years':4, '9 years':9, '7 years':7, '1 year':1, '6 years':6})

# Run it one time on the secons time the all values become NaN.

# To solve this problem, run the code from beginning.
temp_df = df_train.isnull().sum().reset_index()

temp_df['Percentage'] = (temp_df[0]/len(df_train))*100

temp_df.columns = ['Column Name', 'Number of null values', 'Null values in percentage']

print(f"The length of dataset is \t {len(df_train)}")

temp_df
# Let's remove unwanted columns

try:

    df_test.drop(labels=['Loan ID', 'Customer ID'], axis=1, inplace=True)

    df_train.drop(labels=['Loan ID', 'Customer ID'], axis=1, inplace=True)

    

    

except Exception as e:

    pass
sns.countplot(data=df_train, x='Term')

plt.show()
df_train['Term'].fillna(value='Short Term', inplace=True)

df_test['Term'].fillna(value='Short Term', inplace=True)

sns.countplot(data=df_train, x='Home Ownership')

plt.show()
df_train['Home Ownership'].unique()
df_train['Home Ownership'].fillna(value='Home Mortgage', inplace=True)

df_test['Home Ownership'].fillna(value='Home Mortgage', inplace=True)
sns.countplot(data=df_train, x='Purpose')

plt.xticks(rotation=90)

plt.show()
df_train['Purpose'].fillna(value='Debt Consolidation', inplace=True)

df_test['Purpose'].fillna(value='Debt Consolidation', inplace=True)
sns.countplot(data=df_train, x='Loan Status')

plt.xticks(rotation=90)

plt.show()
 # Let us plot the same graph but inter-active this time with the help of plotly.

    

count = df_train['Loan Status'].value_counts().reset_index()

count.iplot(kind='bar', x='index', y='Loan Status', xTitle='Loan Status', yTitle='Frequency',

           color='deepskyblue', title='Fully Paid VS Charged off')
df_train['Loan Status'].fillna(value='Fully Paid', inplace=True)
# Let us Import the Important Libraries  to train our Model for Machine Learning 

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder, OneHotEncoder # To deal with Categorical Data in Target Vector.

from sklearn.model_selection import train_test_split  # To Split the dataset into training data and testing data.

from sklearn.model_selection import cross_val_score   # To check the accuracy of the model.

from sklearn.preprocessing import Imputer   # To deal with the missing values

from sklearn.preprocessing import StandardScaler   # To appy scaling on the dataset.
# Convert DataFrame into array.

x_train = df_train.drop(labels='Loan Status', axis=1).values

y_train = df_train['Loan Status'].values

x_test = df_test.values
imputer = Imputer()

x_train[:, [0,2,3,4,7,8,9,10,11,12,13,14,15]]  = imputer.fit_transform(x_train[:, [0,2,3,4,7,8,9,10,11,12,13,14,15]])

x_test[:, [0,2,3,4,7,8,9,10,11,12,13,14,15]]  = imputer.fit_transform(x_test[:, [0,2,3,4,7,8,9,10,11,12,13,14,15]])

labelencoder_x = LabelEncoder()

x_train[:, 1 ] = labelencoder_x.fit_transform(x_train[:,1 ])

x_train[:, 5 ] = labelencoder_x.fit_transform(x_train[:,5 ])

x_train[:, 6 ] = labelencoder_x.fit_transform(x_train[:,6 ])



#this is need to done when we have more than two categorical values.

onehotencoder_x = OneHotEncoder(categorical_features=[1,5,6]) 

x_train = onehotencoder_x.fit_transform(x_train).toarray()



# Let's apply same concept on test set.

x_test[:, 1 ] = labelencoder_x.fit_transform(x_test[:,1 ])

x_test[:, 5 ] = labelencoder_x.fit_transform(x_test[:,5 ])

x_test[:, 6 ] = labelencoder_x.fit_transform(x_test[:,6 ])



onehotencoder_x = OneHotEncoder(categorical_features=[1,5,6]) 

x_test = onehotencoder_x.fit_transform(x_test).toarray()
labelencoder_y=LabelEncoder()

y_train = labelencoder_y.fit_transform(y_train)
sc_X=StandardScaler()

x_train=sc_X.fit_transform(x_train)

x_test = sc_X.fit_transform(x_test)
from sklearn.decomposition import PCA

pca = PCA(n_components=None)

x_train = pca.fit_transform(x_train)

x_test = pca.fit_transform(x_test)

explained_variance = pca.explained_variance_ratio_

explained_variance
pca = PCA(n_components=25)

x_train = pca.fit_transform(x_train)

x_test = pca.fit_transform(x_test)
 # Apply Logistic regression

    # First step is to train our model .



classifier_logi = LogisticRegression()

classifier_logi.fit(x_train,y_train)



# Let us check the accuracy of the model with k-cross validation.

accuracy = cross_val_score(estimator=classifier_logi, X=x_train, y=y_train, cv=10)

print(f"The accuracy of the Logistic Regressor Model is \t {accuracy.mean()}")

print(f"The deviation in the accuracy is \t {accuracy.std()}")
# Train Model

"""classifier_ran = RandomForestClassifier()

classifier_ran.fit(x_train,y_train)



# Check the accuracy and deviation in the accuracy

accuracy = cross_val_score(estimator=classifier_ran, X=x_train, y=y_train, cv=10)

print(f"The accuracy of the Random Forest Model is \t {accuracy.mean()}") 

print(f"The deviation in the accuracy is \t {accuracy.std()}")"""

# Here we are getting the accuracy of 79%. 

print("In Random Forest Model we are getting the accuracy of 79%")
print(np.unique(y_train))

print(y_train[:10])

print("Here 1 indicates 'Fully Paid'. And 0 indicates 'Charged Off' ")
y_pred = classifier_logi.predict(x_test)



# Let us convert 1 and 0 into Fully Paid and Charged off respectively

y_pred = list(map(lambda x: 'Fully Paid' if x==1 else 'Charged Off' ,y_pred))

y_pred = np.array(y_pred)

y_pred[:5]