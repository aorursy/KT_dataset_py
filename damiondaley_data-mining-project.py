# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import seaborn as sns
from sklearn import linear_model
path_to_file="../input/googleplaystore.csv"
data=pd.read_csv(path_to_file,encoding='utf-8')
##checking for all null values in dataset
missing_data_results =data.isnull().sum()
print(missing_data_results)

#loops through dataset and delete rows where column values is null
data =data.dropna()
data.isnull().sum()
#Below we are using the regex \D to remove any non-digit characters
data['Installs']=data['Installs'].replace(regex=True,inplace=False,to_replace=r'\D',value=r'')
#data['Installs']=data['Installs'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
#data.Installs

data['Installs']
np.sort(data.Installs)
#for col in data.columns:
  #  data[col]=np.sort(data[col].values)
data.shape
dupes=data.duplicated()
sum(dupes)

data=data.drop_duplicates()
data.shape
data.dtypes
    
data['Price']=data['Price'].replace(regex=True,inplace=False,to_replace=r'\D',value=r'')
#converting installs and Price to appropriate data types
data["Installs"] = pd.to_numeric(data["Installs"])
data["Reviews"] = pd.to_numeric(data["Reviews"])
data["Price"] = pd.Float64Index(data["Price"])
data.dtypes
data.columns =data.columns.str.replace(' ', '_')
#Apps available based on Content rating
plt.figure(figsize=(10,10))
sns.countplot(x='Content_Rating',data=data,)
plt.xticks(rotation=45)
plt.title("Number of Apps available based on Content rating")


#data['Category'].value_counts()
plt.figure(figsize=(12,12))
data['Category'].value_counts().plot(kind='bar',title='Distribution of Categories')
plt.xlabel('Categories')
plt.ylabel('Number of Apps')



plt.figure(figsize=(12,12))
sns.barplot(x='Installs',y='Category',data=data,ci=None)
plt.title("Number of Apps installed based on Category")

data['Type'].value_counts().plot(kind='bar',title='Distribution App Types')
plt.xlabel('Type of Apps')
plt.ylabel('Count')

find=((data.Installs.values >=10000)& (data.Installs.values <=10000000))
data1 = data[find]
data1
#data.hist(column= 'Installs')
data1.hist(column= 'Installs')
len(data1.Installs.values)
data1.Reviews
data1.Reviews=pd.qcut(data1.Reviews,20)
tree_data = data1[['Installs','Category','Type','Reviews']]
tree_data
data1.Installs.value_counts()
tree_data['Installs'] = pd.cut(tree_data['Installs'], [9999
,50000
,100000
,500000
,1000000
,5000000
,10000000
                                              ])
tree_data.Installs.value_counts()
tree_data.Installs.value_counts()
# Encoder function.....transforming
def encoder(dataset):
    from sklearn.preprocessing import LabelEncoder
    #dictionary to store values
    encoder = {}
    for column in dataset.columns:
        # Only creating encoder for categorical data types
      #  if not np.issubdtype(dataset[column].dtype, np.number) and column != 'Installs':
            encoder[column]= LabelEncoder().fit(dataset[column])
            #returning the dictionary with values
    return encoder
tree_data
#transforming tree data
encoded_labels = encoder(tree_data)
print("Encoded Values for each Label")
print("="*32)
for column in encoded_labels:
    print("="*32)
    print('Encoder(%s) = %s' % (column, encoded_labels[column].classes_ ))
    print(pd.DataFrame([range(0,len(encoded_labels[column].classes_))], columns=encoded_labels[column].classes_, index=['Encoded Values']  ).T)
data1.Installs.value_counts()
transformed_data= tree_data.copy()
for col in transformed_data.columns:
    if col in encoded_labels:
       transformed_data[col] = encoded_labels[col].transform(transformed_data[col])
print("Transformed data set with category and type encoded")
print("="*32)
transformed_data
from sklearn.model_selection import train_test_split
#Seperate our data into independent X and dependent Y 
X_data = transformed_data[['Category','Type']]
Y_data= transformed_data['Installs']
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.30)
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
# creating multinomial model since we have more than one predictor then fit training data.
regr = linear_model.LogisticRegression(solver='newton-cg')
#regr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(pd.DataFrame(X_train),Y_train)
#regr = GaussianNB()
regr.fit(pd.DataFrame(X_train),Y_train)
#given a trained model, we are predicting the label of a new set of X test data.
Prediction = regr.predict(pd.DataFrame(X_test))
transformed_data['Installs'].value_counts()
print(Prediction)
# The coefficient of our determinant(x)
print('Coefficients: \n', regr.coef_)
regr.intercept_
from sklearn.metrics import r2_score
# Use score method to get accuracy of model
print('Variance score:%2f'% r2_score(Y_test,Prediction)) 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
cm = metrics.confusion_matrix(Y_test,Prediction)
print(cm)
cm.shape
plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix', size = 15)
plt.colorbar()
plt.xticks(Prediction)
plt.yticks(Y_test)
plt.tight_layout()
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
width,height = cm.shape
for x in range(width):
 for y in range(height):
  plt.annotate(str(cm[x][y]), xy=(y, x), 
  horizontalalignment='center',
  verticalalignment='center')
data1
plt.scatter(transformed_data.Category,transformed_data.Installs)
plt.show()

np.corrcoef(transformed_data.Type,transformed_data.Installs)

plt.scatter(transformed_data.Type,transformed_data.Installs)
plt.show()
np.corrcoef(transformed_data.Type,transformed_data.Installs)
from sklearn.tree import DecisionTreeClassifier

from sklearn import tree
# Create the classifier with a maximum depth of 2 using entropy as the criterion for choosing most significant nodes
# to build the tree
clf = DecisionTreeClassifier(criterion='entropy',min_samples_split=2)
# Hint : Change the max_depth to 10 or another number to see how this affects the tree
clf.fit(X_train, Y_train)
pd.DataFrame([ "%.2f%%" % perc for perc in (clf.feature_importances_ * 100)
], index = X_data.columns, columns = ['Feature Significance in Decision Tree'])
import graphviz
Y_data
dot_data = tree.export_graphviz(clf,out_file=None,

feature_names=X_data.columns,
class_names= None,
filled=True, rounded=True, proportion=True,
node_ids=True, #impurity=False,
special_characters=True)
graph = graphviz.Source(dot_data)

graph
tree.export_graphviz(clf,out_file='tree.dot') 
corrmat = transformed_data.corr()
#f, ax = plt.subplots()
p =sns.heatmap(corrmat, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
transformed_data['Reviews'].corr(transformed_data['Installs'])
X_data1 = transformed_data['Reviews']
Y_data1 = transformed_data['Installs']
from sklearn.model_selection import train_test_split

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_data1, Y_data1, test_size=0.30)
reg1 = linear_model.LinearRegression()
reg1.fit(pd.DataFrame(X_train1),y_train1)
Prediction1 = reg1.predict(pd.DataFrame(X_test1))
y_test1.index
Prediction1[:12]
reg1.coef_
reg1.intercept_
reg1.score(pd.DataFrame(X_test1),y_test1)
plt.scatter(X_test1,y_test1,  color='black')
plt.plot(X_test1,Prediction1,color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
import seaborn as sns
sns.set(style="whitegrid")
# Plot the residuals after fitting a linear model
sns.residplot(X_train1, y_train1, lowess=True, color="b")
install = 0.2*4 -0.354
install
cluster_data = transformed_data[['Reviews','Installs']]
cluster_data.head(50)
cluster_data.plot(kind='scatter',x='Reviews',y='Installs')
# Is there any missing data
missing_data_results = cluster_data.isnull().sum()
print(missing_data_results)
data_values = cluster_data.iloc[ :, :].values
data_values
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
mms = MinMaxScaler()
mms.fit(data_values)
data_transformed = mms.transform(data_values)
Sum_of_squared_distances = []
K = range(1,15)
for i in K:
    km = KMeans(n_clusters=i)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('WCSS')
plt.title('Computing WCSS for KMeans++')
plt.xlabel("Number of clusters")
plt.show()
kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300)
cluster_data["cluster"] = kmeans.fit_predict( data_values )
cluster_data
#viewing amount of elements in clusters
cluster_data['cluster'].value_counts()

import seaborn as sns
sns.set(color_codes=True)

cluster_data['cluster'].value_counts().plot(kind='bar',title='Distribution of Apps')
grouped_cluster_data = cluster_data.groupby('cluster')
grouped_cluster_data
grouped_cluster_data.plot(subplots=True)
sns.pairplot(cluster_data,hue="cluster")