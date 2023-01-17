import pandas as pd
import sklearn.datasets as skl_datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection as sklms
import sklearn.linear_model as skllm
from sklearn.neighbors import KNeighborsClassifier
ds_iris = skl_datasets.load_iris()
print(ds_iris.DESCR)
X = ds_iris.data
features = ds_iris.feature_names
y = ds_iris.target
target = ds_iris.target_names
print(X.shape)
X[:5,:]
features
features = ['sepal-length','sepal-width','petal-length','petal-width']
features
print(y.shape)
y[:5]
target
# we convert numpy array to pandas dataframe
df_X = pd.DataFrame(X)
# we reshape the target variable to make it same number of rows as in X and 
# 1 column
y = y.reshape(-1,1)
# we convert y also to a dataframe
df_y = pd.DataFrame(y)
# finally we merge X and y together. axis=1 means concatenate column wise
df = pd.concat([df_X,df_y], axis=1)
# now we update feature and target names to the merged dataframe
features.append("class")
df.columns = features
# Finally, we look at the head of our monster :-)
df.head()
# Looks complete
# Did you notice first column contains integers. This is index of your table
# or dataframe
df.index
df.describe()
# Looking at data, 
# I remember the song from Michael Jackson "The way you make me feel"
# You get a feeler for your data
# Each plant has a sepal and a petal
# Both have length and width
# We use these characteristics to classify each plant
# count - tells me that there no missing values or null observations in
# each column
# Looking at mean of Sepal length tells you that 
# it is the longest among all features
# You look at spread now and find petal length to have maximum variance
# you also get to know max and min of each feature
# also percentiles for each column
pd.pivot_table(data=df,index='class', aggfunc=[np.mean, np.median])
pd.pivot_table(data=df,index='class', aggfunc=[len, np.std])
# Min, Max, Mean and Standard Deviation gives you a good feel about your data
# This is where ART comes into the picture. You can make your machine learning
# model more beautiful by look at all features and imagining what other
# features can be derived from available ones
df.groupby('class').agg([min, max, np.mean, np.std]).round(2)
# we also look at data types of all features and target
df.info()
# data types looks ok. in case you dont feel ok, you need to convert them to 
# appropriate types like int, float, String, date time, etc.
# this will be covered in later articles
# first plot we look at is box plot for all the numerical features
sns.boxplot(df[['sepal-length','sepal-width','petal-length','petal-width']])
plt.show()
# time of finding correlation between features and target, also within features
# why within features is a topic for later articles
sns.pairplot(df, hue='class')
plt.show()
# Prepare X and y
array = df.values
X = array[:,0:4]
y = array[:,4]
# Split into training and test dataset. 0.3 means 70% is training observations
validation_size = 0.30
# This is the random seed. So that when you run this code, we are on same 
# wavelength to discuss the model and results
seed = 7
# We are using Kfold cross validation to split the dataset and perform training
# and testing of the model. I will cover this later.
no_of_splits = 10
# This is your algorithm and you instantiate and pass it along with data
model = skllm.LogisticRegression()

# Used for splitting the dataset. Will be expanded in later article
kfold = sklms.KFold(n_splits=no_of_splits, random_state=seed)

# Here goes everything into the frying pan
X_train, X_test, Y_train, Y_test = sklms.train_test_split(
    X, y, test_size=validation_size, random_state=seed)

# and here comes the output or prediction. in this case we measure accuracy
cv_results = sklms.cross_val_score(model, X_train, Y_train, \
                             cv=kfold, scoring='accuracy')
msg = "%s: %f (%f)" % ("Logistic regression:", cv_results.mean(), cv_results.std())
print(msg)
# Here we use another algorithm to predict
knn = KNeighborsClassifier()
# we train our model
knn.fit(X_train, Y_train)
# predict on our test observations
predictions = knn.predict(X_test)
# and here are our predictions
print(predictions)