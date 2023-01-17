# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

#import plotly.graph_objs as go

#import plotly.offline as py

from sklearn.model_selection import train_test_split

from sklearn import linear_model

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_path = '../input/suicide_rates_1.csv'

data = pd.read_csv(data_path)

data.head(100)
df = pd.DataFrame(data)

df=df.rename(columns = {'suicides/100k pop':'suicides_100k_pop','country-year':'country_year','HDI for year':'HDI_for_year'})

data = df

data
data.shape
data.columns
data.describe()
data.describe(include = ['object'])
#data = data.drop("gdp_for_year",axis=0)
data_copy = data.copy()

df = pd.DataFrame(data_copy)

data_copy
data_copy.dtypes
data_copy.isnull().sum()
df['country'].unique()
help(len)
len(df['country'].unique().tolist())
def create_label_encoder_dict(df):

    from sklearn.preprocessing import LabelEncoder

    label_encoder_dict = {}

    for column in df.columns:

        # Only create encoder for categorical data types

        if not np.issubdtype(df[column].dtype, np.number) and column != 'country_year':

            label_encoder_dict[column]= LabelEncoder().fit(df[column])

    return label_encoder_dict
label_encoders = create_label_encoder_dict(data_copy)

print("Encoded Values for each Label")

print("="*32)

for column in label_encoders:

    print("="*32)

    print('Encoder(%s) = %s' % (column, label_encoders[column].classes_ ))

    print(pd.DataFrame([range(0,len(label_encoders[column].classes_))], columns = label_encoders[column].classes_, index=['Encoded Values'] ).T)
#Apply each encoder to the data set to obtain transformed values

copy_suicide_rate = data_copy.copy() # create another copy of data set 

for column in copy_suicide_rate.columns:

    if column in label_encoders:

        copy_suicide_rate[column] = label_encoders[column].transform(copy_suicide_rate[column])

print("Transformed data set")

print("="*32)

copy_suicide_rate
copy_suicide_rate.columns
#Made a copy of the transformed Dataset to perform Regression on 

regData = copy_suicide_rate
#Seperate our data into dependent (Y) and independent X values

XX_data = regData[['country', 'year', 'sex', 'suicides_no', 'population',

       'suicides_100k_pop']]

YY_data = regData['generation'] 
#Splitting Data using a 70/30 split

XX_train, XX_test, yy_train,yy_test = train_test_split(XX_data,YY_data, test_size = 0.30)
regres = linear_model.LinearRegression()

regres.fit(XX_train,yy_train)
regres.coef_
XX_train.columns


print("Regression Coefficients")

pd.DataFrame(regres.coef_,index= XX_train.columns,columns=["Coefficient"])

regres.intercept_

# Make predictions using the testing set

test_predicted = regres.predict(XX_test)

test_predicted
from sklearn.metrics import mean_squared_error, r2_score
# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(yy_test, test_predicted))
# R squared

print('Variance score: %.2f' % r2_score(yy_test, test_predicted))
## Tool used to compress the data into two dimensions

from sklearn.decomposition import PCA

pca = PCA(n_components=1)


pca.fit(regData[XX_train.columns])
pca.components_
pca.n_features_
pca.n_components_
XX_test
XX_reduced = pca.transform(XX_test)

XX_reduced
plt.scatter(XX_reduced, yy_test,  color='black')
plt.scatter(XX_reduced, yy_test,  color='black')

plt.plot(XX_reduced, test_predicted, color='blue',linewidth=1)





plt.show()

#separate our data into dependent (Y) and independent (X) variables

X_data = copy_suicide_rate[['country','year','sex','age','suicides_100k_pop','population', 

   'gdp_per_capita','generation']]

y_data = copy_suicide_rate['suicides_no']
X_data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.30)
#Decision Tree Algorithm

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree
clf = DecisionTreeClassifier(max_depth=2, criterion='entropy')
clf.fit(X_train, y_train)
y_data
pd.DataFrame([ "%.2f%%" % perc for perc in (clf.feature_importances_ * 100)

], index = X_data.columns, columns = ['Feature Significance in Decision Tree'])
import graphviz
dot_data = tree.export_graphviz(clf,out_file = None,

                               feature_names = X_data.columns,

                            class_names = 'suicides_no',

                            filled=True, rounded=True,proportion=True,

                                node_ids=True,

                             special_characters=True)
graph = graphviz.Source(dot_data) 

graph
#K-Means Algorithm

from sklearn.cluster import KMeans
copy_suicide_rate.columns
cluster_data = copy_suicide_rate[['suicides_no', 'population']]

cluster_data.head()
cluster_data.plot(kind='scatter', x='population', y='suicides_no',figsize=(15, 10))
missing_data_results = cluster_data.isnull().sum()

print(missing_data_results)
data_values = cluster_data.iloc[ :, :].values

data_values
wcss = []

for i in range( 1, 15 ):

    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300) 

    kmeans.fit_predict( data_values )

    wcss.append( kmeans.inertia_ )

    

plt.plot( wcss, 'ro-', label="WCSS")

plt.title("Computing WCSS for KMeans++")

plt.xlabel("Number of clusters")

plt.ylabel("WCSS")

plt.show()
kmeans = KMeans(n_clusters=10, init="k-means++", n_init=10, max_iter=300) 

cluster_data["cluster"] = kmeans.fit_predict( data_values )

cluster_data
cluster_data['cluster'].value_counts()
cluster_data['cluster'].value_counts().plot(kind='bar',title='Distribution of Suicide Rates',figsize=(15, 10))