# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing all the necessary libraries

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(color_codes=True)

import warnings # to ignore warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC      
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('../input/vehicle/vehicle.csv')

df.head(10)  # High level observation of dataset
print("The dataframe has {} rows and {} columns".format(df.shape[0],df.shape[1]),'\n\n')

df.info()
df.isnull().sum() # Null value check
# Five point summary of the numerical attributes
df.describe().T
# Distribution of target column
print(df['class'].value_counts(),'\n')
sns.countplot(df['class']);
#instead of dropping the rows, lets replace the missing values with median value. 
df.median()
# replace the missing values with median value.
# we do not need to specify the column names below
# every column's missing value is replaced with that column's median respectively
df = df.fillna(df.median())    # The fillna() function is used to fill NaN values using the specified method
df.isna().apply(pd.value_counts)   # checking null values now
### Null values are treated
df.duplicated().sum()  # no duplicate values found in the dataset
# Optional, we can implement without replacing the classes with numerical values also
replace_dependent= {'class' : {'car': 0, 'bus': 1, 'van': 2} }

df = df.replace(replace_dependent)
df['class'].value_counts()
df.info()
k=1
plt.figure(figsize=(20,30))

# using for loop to iterate over all the columns in the dataframe and plot the histogram of those 

for col in df.columns[0:18]: # iterating columns except dependent column
    plt.subplot(5,4,k)
    sns.distplot(df[col],color='black')
    k=k+1
k=1
plt.figure(figsize=(20,30))

# using for loop to iterate over all the columns in the dataframe and plot the boxplot of those 
# as we can observe outliers easily in boxplot

for col in df.columns[0:18]: # iterating columns except dependent column
    plt.subplot(5,4,k)
    sns.boxplot(y=df[col],color='black')
    k=k+1
# kde plots to show the distribution of the all the variables with respect to dependent variable
k=1
plt.figure(figsize=(20,30))
for col in df.columns[0:18]:
    plt.subplot(5,4,k)
    sns.kdeplot(df[df['class']==0][col],color='red',label='car',shade=True)
    sns.kdeplot(df[df['class']==1][col],color='blue',label='bus',shade=True)
    sns.kdeplot(df[df['class']==2][col],color='yellow',label='van',shade=True)
    plt.title(col)
    k=k+1
sns.pairplot(df,diag_kind='kde')
fig= plt.subplots(figsize=(20, 10))
sns.heatmap(df.corr(),annot=True, linewidth = 0.2)
correlation_values=df.corr()['class']
pd.DataFrame(correlation_values.sort_values(ascending=False))
x = df.iloc[:,0:18]  # independent variables
y = df['class']      # target variable

# Scaling the data using zscore technique as the predictor values has different scale
from scipy.stats import zscore
XScaled=x.apply(zscore)
XScaled.head()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(XScaled, y, test_size=0.2, random_state = 56)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
fig= plt.subplots(figsize=(20, 10))
sns.heatmap(XScaled.corr(),annot=True, linewidth = 0.2)
# instantiating the model
svm_model = SVC()

# fitting the model
svm_model.fit(x_train, y_train)

# score on unseen data
accuracy = svm_model.score(x_test,y_test)
print(accuracy*100)
scores_df = pd.DataFrame({ 'Model' : 'SVM',  'Accuracy' : [accuracy*100] })
scores_df
# prediction using test data
y_pred = svm_model.predict(x_test)

# generating classification report of actual and predicted values
print(metrics.classification_report(y_test,y_pred))
# confusion matrix of actual and predicted values
metrics.confusion_matrix(y_test,y_pred)
# number of splits (25)
num_folds = 25

# initialising kfold object
kfold = KFold(n_splits = num_folds, random_state = 56)

# specifying the model to perform cross validation
model = SVC()

# noting accuracy scores of all the 25 split runs
scores = cross_val_score(model, XScaled, y, cv = kfold)

# printing all the 25 scores
print(scores)
print('')

# here we are getting average accuracy with standard deviation for range estimate
print('Overall Accuracy : {:.2f}% ({:.2f}%)'.format(scores.mean()*100.0, scores.std()*100.0))
scores_df1 = pd.DataFrame({'Model': ['SVM(cross_val)'], 'Accuracy' : [scores.mean()*100]})
scores_df = pd.concat([scores_df,scores_df1]).drop_duplicates()
scores_df
covMatrix = np.cov(XScaled,rowvar=False)
pd.DataFrame(covMatrix)
 # Covariance is the direction of the linear relationship between variables.
pca = PCA()
pca.fit(XScaled)
# Eignen values
pca.explained_variance_
# Eigen Vectors
pd.DataFrame(pca.components_)
k = 1
total = []
for i in pca.explained_variance_ratio_*100:
    print('Variance explained by Principle Component',k,'is : {:.2f}%'.format(i))
    k+=1
    total.append(i)
print('\nTotal variance explained by all the principle components:',sum(total),'%')
# Implementing scree plot 
plt.figure(figsize=(16 , 5))
plt.bar(range(1, 19), pca.explained_variance_ratio_, label = 'Individual explained variance',color='lightblue',edgecolor='black')
plt.step(range(1, 19), np.cumsum(pca.explained_variance_ratio_),where='mid', label = 'Cumulative explained variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.legend(loc = 'best')
plt.show()
# NOTE - we are generating only 7 PCA dimensions (dimensionality reduction from 18 to 7)

pca2 = PCA(n_components=7)  # here you can notice we are specifying 7 PC components in the parameter called n_components
pca2.fit(XScaled)
# Eigen Vectors (transformed into 7 dimensions)
pd.DataFrame(pca2.components_).T
# Eigen Values (only 7)
pca2.explained_variance_
# Percentage of variance explained by 7 PC components
sum(pca2.explained_variance_ratio_*100)
# Now Implementing scree plot on 8 variables
plt.figure(figsize=(16 , 5))  # size of the plot

# bar plot
plt.bar(range(1, 8), pca2.explained_variance_ratio_, label = 'Individual explained variance',color='lightblue',edgecolor='black')

# step plot on bars which is a cummulative sum of the variance explained by 7 pc components
plt.step(range(1, 8), np.cumsum(pca2.explained_variance_ratio_),where='mid', label = 'Cumulative explained variance',color = 'black')

plt.ylabel('Explained Variance Ratio')  # x axis label
plt.xlabel('Principal Components')      # y axis label
plt.legend()
# #transforming the impute raw data which is in 18 dimension into 7 new dimension with pca
pca_transformed =  pca2.transform(XScaled)
# checking the shape of pca_transformed data
pca_transformed.shape
# Visualising PCA dimensions with pair panel
sns.pairplot(pd.DataFrame(pca_transformed),diag_kind = 'kde')
# will see the shape of original train and test dataset

print('original data shape')
print('shape of x_train',x_train.shape)
print('shape of y_train',y_train.shape)
print('shape of x_test',x_test.shape)
print('shape of y_test',y_test.shape)

# split the transformed pca data
pca_x_train, pca_x_test, pca_y_train, pca_y_test = train_test_split(pca_transformed, y, test_size = 0.2, random_state = 56)


# Shape of new train and test data
print('*** Transformed data using pca ***')
print('   shape of pca_x_train:',pca_x_train.shape)
print('   shape of pca_y_train:',pca_y_train.shape)
print('   shape of pca_x_test:',pca_x_test.shape)
print('   shape of pca_y_test:',pca_y_test.shape)

# creating a dataframe with the new dataset
### pca_transformed is the new dataset with 7 principle components
pca_transformed = pd.DataFrame(pca_transformed,columns = df.columns[0:7])

# shape of the dataframe
print(pca_transformed.shape)  

# displaying head of the dataframe
pca_transformed.head(10)
# instanstiating the object of SVM model / building the svm model using principle components instead of original data
svm_pca = SVC()

# fitting the model on new data
svm_pca.fit(pca_x_train, pca_y_train)
# score of test data
print('Accuracy score of SVM model after reducing dimensions :',svm_pca.score(pca_x_test,pca_y_test),'\n\n')

# prediction using pca test data
svm_pca_pred = svm_pca.predict(pca_x_test)

# generating classification report of actual and predicted values
print(metrics.classification_report(pca_y_test, svm_pca_pred))

# confusion matrix of actual and predicted values
print('\n Confusion matrix:\n',metrics.confusion_matrix(pca_y_test, svm_pca_pred))
scores_df1 = pd.DataFrame({'Model': ['SVM with PCA'], 'Accuracy' : [svm_pca.score(pca_x_test,pca_y_test)*100]})
scores_df = pd.concat([scores_df,scores_df1]).drop_duplicates()
scores_df
# number of splits
num_folds = 25

# initialising kfold object
kfold = KFold(n_splits = num_folds, random_state = 56)

# specifying the model to perform cross validation
model = SVC()

# noting accuracy scores of all the 25 split runs
scores = cross_val_score(model, pca_transformed, y, cv = kfold)

# printing all the 25 scores
print(scores)
print('')

# here we are getting average accuracy with standard deviation for range estimate
print('Overall Accuracy : {:.2f}% ({:.2f}%)'.format(scores.mean()*100.0, scores.std()*100.0))
scores_df1 = pd.DataFrame({'Model': ['SVM(cross_val) with PCA'], 'Accuracy' : [scores.mean()*100]})
scores_df = pd.concat([scores_df,scores_df1]).drop_duplicates()
scores_df
scores_df
fig = plt.figure(figsize=(9,5))
plt.title('Accuracy values for various techniques')

ax = sns.barplot(scores_df['Model'],scores_df['Accuracy'],color='lightblue');

for p in ax.patches:
    ax.annotate('{:.1f}%'.format(p.get_height()), 
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha = 'center', va = 'center', 
                    xytext = (0, -12), 
                    textcoords = 'offset points') # used annotation to show the percentage of accuracy
