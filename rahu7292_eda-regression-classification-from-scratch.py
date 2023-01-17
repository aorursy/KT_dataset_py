# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot, download_plotlyjs

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))

print()

print("The files in the dataset is:-- ")

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Importing the dataset

df = pd.read_csv('../input/BlackFriday.csv')
# Let us check the Top 5 entries in the dataset.

df.head()
# Basic information 

df.info()
# Let us check the Number of Null values and it's Percentage .

temp_df = df.isnull().sum().reset_index()

temp_df['Percentage of Null Values'] = temp_df[0]/len(df)*100

temp_df.columns = ['Column Name', 'Number of Null Values','Percentage of Null Values']

temp_df
# Removing Product_Category_3 column

df.drop(labels='Product_Category_3', axis=1, inplace=True)
# Let us fill the Null values in Product_Categorical_2 column

# First let us check the product which come maximum number of times in this column.

new_df = df['Product_Category_2'].value_counts().reset_index()



new_df.iplot(kind='bar', x='index',y='Product_Category_2', title='Frequency of values in Product_Category_2 Column',

            xTitle='Product Category', yTitle='Frequency', color='deepskyblue')

# Filling Null values.

df['Product_Category_2'].fillna(value=8, inplace=True)

df['Product_Category_2'].unique()
gender = df['Gender'].value_counts().reset_index()

gender.columns = ['Gender', 'Count']

gender.iplot(kind='pie', labels='Gender', values='Count', hole=0.2, pull=0.2, title='Ratio of Male and Female')

# Interactive plot with plotly.
age = df['Age'].value_counts().reset_index()

age.columns = ['Age Group', 'Count']

age.iplot(kind='bar', x='Age Group', y='Count', title='Number of people belongs to different age group',

         xTitle='Age Group', yTitle='Quantity', color='deepskyblue')
try:

    df.drop(labels=['User_ID', 'Product_ID'], inplace=True, axis=1,)

except Exception as e: 

    print("Run from start again.")

df.head()
city = df['City_Category'].value_counts().reset_index()

city.iplot(kind='pie', labels='index', values='City_Category', hole=0.2, pull=0.2,

          title='Percentage of People belongs to different City Category')
temp_df = df.groupby('Gender', axis=0)['Purchase'].sum().reset_index()

temp_df.iplot(kind='bar', x='Gender', y='Purchase', title='Total Purchase amount of Male and Female',

             xTitle='Gender', yTitle='Total Purchase Amount', color='orange')

status = df['Marital_Status'].value_counts().reset_index()

# Converting 0 and 1 into Married and Non Married in status DataFrame

status['index'] = status['index'].apply(lambda x: 'Non Married' if x==0 else 'Married')



status.iplot(kind='pie', labels='index', values='Marital_Status', hole=0.2, pull=0.2,

          title='Ratio of Married people and Non Married people')

status.T
temp_df = df.groupby('Marital_Status', axis=0)['Purchase'].sum().reset_index()

temp_df['Marital_Status'] = temp_df['Marital_Status'].apply(lambda x: 'Non Married' if x==0 else 'Married')



temp_df.iplot(kind='bar', x='Marital_Status', y='Purchase', title='Total Purchase amount of Male and Female',

             xTitle='Marital_Status', yTitle='Total Purchase Amount', color='red')

pro1 = df['Product_Category_1'].reset_index()

pro1.columns = ['index', 'product']



pro2 = df['Product_Category_2'].reset_index()

pro2.columns = ['index', 'product']

product = pd.concat([pro1, pro2] )



product = product['product'].value_counts().reset_index()[:10]

product.iplot(kind='bar', x='index', y='product', title='Top 10 product number which purchase maximum number of times',

             xTitle='Product number', yTitle='Frequency', color='deepskyblue')
rich = df.groupby('City_Category')['Purchase'].sum().reset_index()

rich.sort_values('Purchase',ascending=False, inplace=True)

rich.iplot(kind='pie', labels='City_Category' ,values='Purchase', hole=0.2, pull=0.2,

           title='Richest among A,B,C', )

df.corr().iplot(kind='heatmap', title='HeatMap on Correlation of features ')
# IMporting the useful Machine Learning libraries.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

from sklearn.model_selection import cross_val_score

# Let us create a feature matrix and target vector.

x_train = df.iloc[:,:-1].values      # Feature matrix

y_train = df.iloc[:,-1].values     # Target vector
# Apply LabelEncoder to convert categorical values into 0,1 and so on format.

label_x = LabelEncoder()

x_train[:,0] = label_x.fit_transform(x_train[:,0])

x_train[:,1] = label_x.fit_transform(x_train[:,1])

x_train[:,3] = label_x.fit_transform(x_train[:,3])

x_train[:,4] = label_x.fit_transform(x_train[:,4])



# Apply OneHotEncoder to split the columns which have more than 2 categories.

one_hot = OneHotEncoder(categorical_features=[1,2,3,4,6,7])

x_train = one_hot.fit_transform(x_train).toarray()

print(f"""Now let us see the shape of our x_train matrix.

Shape of x_train matrix =\t {x_train.shape}.

So now we have 73 columns.

Now we will apply Dimensionalty Reduction to reduce the number of features/columns with the help of PCA algorithm. """)
# Scaling the feature matrix.

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)



# Scaling the Target Vector.

sc_y = StandardScaler()

y_train = y_train.reshape(-1,1)    # Converting the 1-D array into 2-array.

y_train = sc_x.fit_transform(y_train)
pca = PCA(n_components=None)

x_train = pca.fit_transform(x_train)

pca.explained_variance_ratio_
pca = PCA(n_components=50)   # Fitting PCA with 50 features.

x_train = pca.fit_transform(x_train)
# Multi-linear regression Model. 

regressor_multi = LinearRegression()

regressor_multi.fit(x_train,y_train)

# Let us check the accuray

accuracy = cross_val_score(estimator=regressor_multi, X=x_train, y=y_train,cv=10)

print(f"The accuracy of the Multi-linear Regressor Model is \t {accuracy.mean()}")

print(f"The deviation in the accuracy is \t {accuracy.std()}")

print()
"""

# SVR 

regressor_svr = SVR(kernel='rbf')

regressor_svr.fit(x_train, y_train)

# Let us check the accuracy

accuracy = cross_val_score(estimator=regressor_svr, X=x_train, y=y_train,cv=10)

print(f"The accuracy of the SVR Model is \t {accuracy.mean()}")

print(f"The deviation in the accuracy is \t {accuracy.std()}")

print()

"""

print("Apply this model, if accuracy is good then use this model")

print("This model takes too much time. ")
df.head()
# Creating Feature matrix and Target vector.

x_train = df.drop(labels='Marital_Status',axis=1).values

y_train = df['Marital_Status'].values
# Apply LabelEncoder to convert categorical values into 0,1 and so on format.

label_x = LabelEncoder()

x_train[:,0] = label_x.fit_transform(x_train[:,0])

x_train[:,1] = label_x.fit_transform(x_train[:,1])

x_train[:,3] = label_x.fit_transform(x_train[:,3])

x_train[:,4] = label_x.fit_transform(x_train[:,4])



# Apply OneHotEncoder to split the columns which have more than 2 categories.

one_hot = OneHotEncoder(categorical_features=[1,2,3,4,5,6])

x_train = one_hot.fit_transform(x_train).toarray()

print(f"""Now let us see the shape of our x_train matrix.

Shape of x_train matrix =\t {x_train.shape}.

So now we have 73 columns.

Now we will apply Dimensionalty Reduction to reduce the number of features/columns with the help of PCA algorithm. """)
sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)
pca = PCA(n_components=None)

x_train = pca.fit_transform(x_train)

pca.explained_variance_ratio_
pca = PCA(n_components=50)   # Fitting PCA with 50 features.

x_train = pca.fit_transform(x_train)
# Apply Logistic regression

# First step is to train our model .



classifier_logi = LogisticRegression()

classifier_logi.fit(x_train,y_train)



# Let us check the accuracy of the model

accuracy = cross_val_score(estimator=classifier_logi, X=x_train, y=y_train, cv=10)

print(f"The accuracy of the Logistic Regressor Model is \t {accuracy.mean()}")

print(f"The deviation in the accuracy is \t {accuracy.std()}")
"""



# Apply SVM with Gaussian kernel

classifier_svm2 = SVC(kernel='rbf', )

classifier_svm2.fit(x_train,y_train)

accuracy = cross_val_score(estimator=classifier_svm2, X=x_train, y=y_train, cv=10)

print(f"The accuracy of the SVM Gaussian kernel Model is \t {accuracy.mean()}")

print(f"The deviation in the accuracy is \t {accuracy.std()}")

"""

print("SVC Model")
"""



# Apply Naive Bayes Model.

# Train Model

classifier_bayes = GaussianNB()

classifier_bayes.fit(x_train,y_train)

# Check the accuracy and deviation in the accuracy

accuracy = cross_val_score(estimator=classifier_bayes, X=x_train, y=y_train, cv=10)

print(f"The accuracy of the Naive Bayes Model is \t {accuracy.mean()}") 

print(f"The deviation in the accuracy is \t {accuracy.std()}")

"""

print("Navie-Bayes Model")