## load all libraries
# load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder as le ,MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import math
import sklearn
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from xgboost import XGBRegressor 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV, cross_val_score, KFold, learning_curve

%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
# preprocessing libraries
os.chdir(r'../input/upvotes-dataset')

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample.csv')

train_data.head(5)
# now exploring data
print('shape of training data is : ',train_data.shape)
print('no.of examples in train data is : {} and in test_data is :{}'.format(train_data.shape[0],test_data.shape[0]))
# finding the no. of null values in both train and test
print(train_data.isnull().sum())
print(train_data.isnull().sum())
# since there are no null values we will check is there any outlier present in data 
# outliers greatly affect the performance of model

# removing outlier from the data using inter - quaartile range method
def outlier_removal(df,attributes,n):
    # take loop through the attributes
    outliers_indices = []
    for col in attributes:
        # calculating inter-quartile range
        Q1 = df[col].quantile(0.25)
        # 3rd quartile (75%)
        Q3 = df[col].quantile(0.75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        index = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        outliers_indices.extend(index)
    # finding index which have more than 2
    outliers_indices = Counter(outliers_indices)
    multiple_outliers = list(k for k,v in outliers_indices.items() if v>n) # n tells the no. of outliers present in an example for removal
    return multiple_outliers   
index_list_count = []
for i in range(len(train_data.columns[2:6])):
    multioutlier_indices = outlier_removal(train_data,train_data.columns[2:6],i)
    print('There are {} no. of examples which have outliers in {} columns '.format(len(multioutlier_indices), i+1))
    index_list_count.append(len(multioutlier_indices))      
    print('the examples for this are',train_data.loc[multioutlier_indices] )      
sns.barplot(x = [1,2,3,4],y = index_list_count)
plt.ylabel('no. of example')
plt.xlabel('no. of columns containing outlier')
plt.title('plot for examples vs outlier columns ')
# concatenate the test data an train data to apply same operations
dataset = pd.concat(objs = [train_data,test_data],axis=0,sort=True).reset_index(drop=True)
dataset.drop('ID',axis=1,inplace=True)  # dropping id to be not used for prediction in any way
print(len(dataset))
dataset
# now checking the scales of train data
train_data.drop('ID',axis=1,inplace=True)
train_data.describe()

# now checking the info 
train_data.info()

## firstly let's plot the correlation of each individual on upvotes 
# note here no preprocessing is done so the correlation might be different

# making heatmap of correlation
sns.heatmap(train_data.iloc[:,1:].corr(),annot = True, fmt = '.2f')
#  Analysis of Reputation
print('max no. of Reputation on entire dataset : ',dataset['Reputation'].max())
print('min no. of Reputation on entire dataset : ',dataset['Reputation'].min())
print('max no. of Reputation on train dataset : ',train_data['Reputation'].max())
print('min no. of Reputation on train dataset : ',train_data['Reputation'].min())
print('max no. of Reputation on test dataset : ',test_data['Reputation'].max())
print('min no. of Reputation on test dataset : ',test_data['Reputation'].min())
# starting with Reputation

# attribute 1 analysis
g = sns.distplot(dataset["Reputation"], color="m", label="Skewness : %.2f"%(dataset["Reputation"].skew()))
g.legend(loc='best')
g = sns.boxplot(dataset['Reputation'])
### huge no. of outliers now trying to convert it to logarathmic scale to see, skweness reduce or not
dataset['Reputation'] = dataset['Reputation'].map(lambda i:np.log(i) if i>0 else 0)
g = sns.distplot(dataset["Reputation"], color="m", label="Skewness : %.2f"%(dataset["Reputation"].skew()))
g.legend(loc='best')

# checking in box plot
g = sns.boxplot(dataset['Reputation'])
#now checking the relation of 
train_data['Reputation'] = train_data['Reputation'].map(lambda i:np.log(i) if i>0 else 0)

# on train data only for 1st feature
g = sns.scatterplot(x = 'Reputation' , y = 'Upvotes' , data = train_data)

#  Analysis of Answers
print('max no. of Answers on entire dataset : ',dataset['Answers'].max())
print('min no. of Answers on entire dataset : ',dataset['Answers'].min())
print('max no. of Answers on train dataset : ',train_data['Answers'].max())
print('min no. of Answers on train dataset : ',train_data['Answers'].min())
print('max no. of Answers on test dataset : ',test_data['Answers'].max())
print('min no. of Answers on test dataset : ',test_data['Answers'].min())
# Attribute no. 2 (Answers)
g = sns.distplot(dataset["Answers"], color="m", label="Skewness : %.2f"%(dataset["Answers"].skew()))
g.legend(loc='best')
g = sns.boxplot(dataset['Answers'])
# Again treating outliers for this column
### huge no. of outliers
dataset['Answers'] = dataset['Answers'].map(lambda i:np.log(i) if i>0 else 0)
g = sns.distplot(dataset["Answers"], color="m", label="Skewness : %.2f"%(dataset["Answers"].skew()))
g.legend(loc='best')


# checking in box plot
g = sns.boxplot(dataset['Answers'])
#plotting the variation of Upvotes vs Answers
#now checking the relation 
train_data['Answers'] = train_data['Answers'].map(lambda i:np.log(i) if i>0 else 0)

# on train data only for 1st feature
g = sns.scatterplot(x = 'Answers' , y = 'Upvotes' , data = train_data)
#  Analysis of views
print('max no. of views on entire dataset : ',dataset['Views'].max())
print('min no. of views on entire dataset : ',dataset['Views'].min())
print('max no. of views on train dataset : ',train_data['Views'].max())
print('min no. of views on train dataset : ',train_data['Views'].min())
print('max no. of views on test dataset : ',test_data['Views'].max())
print('min no. of views on test dataset : ',test_data['Views'].min())
# Attribute no. 2 (Views)
g = sns.distplot(dataset["Views"], color="m", label="Skewness : %.2f"%(dataset["Views"].skew()))
g.legend(loc='best')
### huge no. of outliers now trying to convert it to logarathmic scale to see, skweness reduce or not
dataset['Views'] = dataset['Views'].map(lambda i:np.log(i) if i>0 else 0)
g = sns.distplot(dataset["Views"], color="m", label="Skewness : %.2f"%(dataset["Views"].skew()))
g.legend(loc='best')

# checking in box plot
g = sns.boxplot(dataset['Views'])
# now username it is assumed that it doesn't have any significant effect on upvote but let' see
g = sns.distplot(dataset["Username"], color="m", label="Skewness : %.2f"%(dataset["Username"].skew()))
g.legend(loc='best')

# checking in box plot
g = sns.boxplot(dataset['Username'])
# checking the relation of username with upvotes
username_div = train_data.groupby('Username')
username_div.Upvotes.apply(np.mean)
# on train data 
g = sns.scatterplot(x = 'Username' , y = 'Upvotes' , data = train_data)
# Analysis of attribute tags
g = sns.countplot("Tag", data = dataset)
# checking in box plot
g = sns.boxplot(train_data['Tag'],dataset['Upvotes'])
sns.barplot(train_data['Tag'],dataset['Upvotes'])
# may be it seems right to label encode the dependencies in Tag
dataset['Tag'] = le().fit_transform(dataset['Tag'])
dataset.head(5)
# firstly applying features scaling
sd = MinMaxScaler()
norm1 = sd.fit(dataset[['Answers','Reputation','Username','Views']])
x = norm1.transform(dataset[['Answers','Reputation','Username','Views']])
x.shape
x
dataset[['Answers','Reputation','Username','Views']] = x[:,:]
dataset.describe()
# slicing back to train and test set

y_train = np.array(dataset['Upvotes'],dtype = np.int64)[0:len(train_data)]
x_train = dataset[0:len(train_data)][['Answers','Reputation','Tag','Username','Views']]
x_test = dataset[len(train_data):][['Answers','Reputation','Tag','Username','Views']]
print(x_train.shape,x_test.shape,y_train.shape)
y_train
# again applying PCA with 8 features
# identifying which features adds how much variance to data


covar_matrix = PCA(n_components = 5) #we have 8 numerical features
covar_matrix.fit(dataset[['Answers','Reputation','Tag','Username','Views']])
variance = covar_matrix.explained_variance_ratio_ #calculate variance ratios

var=np.cumsum(np.round(variance, decimals=3)*100)
components = covar_matrix.fit_transform(dataset[['Answers','Reputation','Tag','Username','Views']])
var #cumulative sum of variance explained with [n] features
# checking importance using pvalue 
# Applying p-value to check feature dependence
import statsmodels.api as sm
regressor_OLS = sm.OLS(endog = y_train, exog = x_train).fit()
regressor_OLS.summary()
# making heatmap of correlation
sns.heatmap(pd.concat([x_train,pd.DataFrame(y_train, columns= ['Upvotes'])],axis=1).corr(),annot = True, fmt = '.2f')
index_list_count1 = []
for i in range(len(x_train.columns)):
    multioutlier_indices = outlier_removal(x_train,x_train.columns,i)
    print('There are {} no. of examples which have outliers in {} columns '.format(len(multioutlier_indices), i+1))
    index_list_count1.append(multioutlier_indices)      
    print('the examples for this are',train_data.loc[multioutlier_indices] )      
#x_val1 = x_train.loc[index_list_count1[0]]
#y_val1 = pd.DataFrame(y_train).loc[index_list_count1[0]]
x_train = x_train.drop(index_list_count1[0])
len(x_train)
index_list_count1
y_train = pd.DataFrame(y_train).drop(index_list_count1[0])
len(y_train)
# Applying train - val split

x_train,x_val,y_train,y_val  = train_test_split(x_train,y_train,test_size = 0.2, random_state=42)
x_val
# dropping two columns in all

x_train.drop(['Username','Tag'],axis=1,inplace = True)
x_val.drop(['Username','Tag'],axis=1,inplace = True)
x_test.drop(['Tag','Username'],axis=1,inplace = True)
# using k folds cross validation with 5 splits
kfold = KFold( n_splits = 5)

# taking extra trees as model

ExtC = ExtraTreesRegressor(random_state =2)

ExtC.fit(x_train,y_train)
y_val_predict = ExtC.predict(x_val) 
math.sqrt(mean_squared_error(y_val,y_val_predict))

pred = ExtC.predict(x_test)
pred
y_val_predict
sample_submission['Upvotes'] = pred

