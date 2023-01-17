import numpy as np

import pandas as pd



# Visualization



import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Import

train_df = pd.read_csv('../input/train.csv', header=0)

test_df = pd.read_csv('../input/test.csv', header=0)



#Save the id's for submission file

ID = test_df['Id']

#Drop unnecessary columns

test_df.drop('Id',axis=1,inplace=True)
# Dimensions

print( 'Train: ', train_df.shape)

print( 'Test: ', test_df.shape)
# Data structures

train_df.info()

print('-------------------------------------------')

test_df.info()
# Statistics

train_df.describe()
# class distribution

print( train_df.groupby('MSZoning').size())
train_df['SalePrice'].describe()
# distribution of SalePrice

sns.distplot(train_df['SalePrice'], kde=False)
# Correlation tells relation between two attributes.

# Correlation requires continous data. Hence, ignore categorical data



# Calculates pearson co-efficient for all combinations

data_corr = data.corr()



# Set the threshold to select only highly correlated attributes

threshold = 0.5



# List of pairs along with correlation above threshold

corr_list = []



#Search for the highly correlated pairs

for i in range(0,size): #for 'size' features

    for j in range(i+1,size): #avoid repetition

        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):

            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index



#Sort to show higher ones first            

s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))



#Print correlations and column names

for v,i,j in s_corr_list:

    print ("%s and %s = %.2f" % (cols[i],cols[j],v))



# Strong correlation is observed between the following pairs

# This represents an opportunity to reduce the feature set through transformations such as PCA
# Correlation tells relation between two attributes.

# Correlation requires continous data. Hence, ignore categorical data



# Calculates pearson co-efficient for all combinations

#create a dataframe with only continuous features

data = train_df.iloc[:, :]

size = 38

cols=data.columns 

data_corr = data.corr()



# Set the threshold to select only highly correlated attributes

threshold = 0.5



# List of pairs along with correlation above threshold

corr_list = []



#Search for the highly correlated pairs

for i in range(0,size): #for 'size' features

    for j in range(i+1,size): #avoid repetition

        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):

            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index



#Sort to show higher ones first            

s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))



#Print correlations and column names

for v,i,j in s_corr_list:

    print ("%s and %s = %.2f" % (cols[i],cols[j],v))



# Strong correlation is observed between the following pairs

# This represents an opportunity to reduce the feature set through transformations such as PCA
# numerical features correlations

corr = train_df.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()

plt.figure(figsize=(12, 12))

sns.heatmap(corr, vmax=1, square=True)
cor_dict = corr['SalePrice'].to_dict()

del cor_dict['SalePrice']

print("List the numerical features decendingly by their correlation with Sale Price:\n")

for ele in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):

    print("{0}: \t{1}".format(*ele))

    

print('-----------------------------------------')
print('Over 40% correlation with SalePrice:')

for ele in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):

    if(ele[1] > 0.4 ):

        print("{0}: \t{1}".format(*ele))
sns.lmplot(x = 'OverallQual', y = 'SalePrice', data = train_df)#.describe()
# Skewness of the distribution



#print(train_df.skew())



# Values close to 0 show less ske

# loss shows the highest skew. Let us visualize it
plt.figure(1)

f, axarr = plt.subplots(3, 2, figsize=(10, 9))

price = train_df.SalePrice.values

axarr[0, 0].scatter( train_df.GrLivArea.values, price)

axarr[0, 0].set_title('GrLiveArea')

axarr[0, 1].scatter( train_df.GarageArea.values, price)

axarr[0, 1].set_title('GarageArea')

axarr[1, 0].scatter( train_df.TotalBsmtSF.values, price)

axarr[1, 0].set_title('TotalBsmtSF')

axarr[1, 1].scatter( train_df['1stFlrSF'].values, price)

axarr[1, 1].set_title('1stFlrSF')

axarr[2, 0].scatter( train_df.TotRmsAbvGrd.values, price)

axarr[2, 0].set_title('TotRmsAbvGrd')

axarr[2, 1].scatter( train_df.MasVnrArea.values, price)

axarr[2, 1].set_title('MasVnrArea')

f.text(-0.01, 0.5, 'Sale Price', va='center', rotation='vertical', fontsize = 12)

plt.tight_layout()

plt.show()
fig = plt.figure(2, figsize=(9, 7))

plt.subplot(211)

plt.scatter(train_df.YearBuilt.values, price)

plt.title('YearBuilt')



plt.subplot(212)

plt.scatter(train_df.YearRemodAdd.values, price)

plt.title('YearRemodAdd')



fig.text(-0.01, 0.5, 'Sale Price', va = 'center', rotation = 'vertical', fontsize = 12)



plt.tight_layout()
print(train_df.select_dtypes(include=['object']).columns.values)
plt.figure(figsize = (12, 6))

sns.boxplot(x = 'Neighborhood', y = 'SalePrice',  data = train_df)

xt = plt.xticks(rotation=45)
plt.figure(figsize = (12, 6))

sns.countplot(x = 'Neighborhood', data = train_df)

xt = plt.xticks(rotation=45)
#One-hot encoding converts an attribute to a binary vector



#Variable to hold the list of variables for an attribute in the train and test data

labels = []



for i in range(0,split):

    train = train_df[cols[i]].unique()

    test = test_df[cols[i]].unique()

    labels.append(list(set(train) | set(test)))    



del test_df



#Import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



#One hot encode all categorical attributes

cats = []

for i in range(0, split):

    #Label encode

    label_encoder = LabelEncoder()

    label_encoder.fit(labels[i])

    feature = label_encoder.transform(dataset.iloc[:,i])

    feature = feature.reshape(dataset.shape[0], 1)

    #One hot encode

    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))

    feature = onehot_encoder.fit_transform(feature)

    cats.append(feature)



# Make a 2D array from a list of 1D arrays

encoded_cats = numpy.column_stack(cats)



# Print the shape of the encoded data

print(encoded_cats.shape)



#Concatenate encoded attributes with continuous attributes

dataset_encoded = numpy.concatenate((encoded_cats,dataset.iloc[:,split:].values),axis=1)

del cats

del feature

del dataset

del encoded_cats

print(dataset_encoded.shape)