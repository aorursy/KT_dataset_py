#Import the imported libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Read the data

train = pd.read_csv("../input/train.csv")



#Check the data



print(train.head())

print(train.shape)
notnullcount = train.count()

# List the columns with more than 30 % missing values

nullmorethan30 = [n for n in notnullcount if n < 0.3 * train.shape[0]]

removablecolumns =[]

for v in nullmorethan30:

    colr = notnullcount[notnullcount == v].index[0]

    removablecolumns.append(colr)
train = train.drop(removablecolumns,1)    

import numpy as np



trainnew = train

columns = trainnew.columns

for col in columns:

    if(trainnew[col].dtype == np.dtype('O')):

        trainnew[col] = trainnew[col].fillna(trainnew[col].value_counts().index[0])

        #print(trainnew[col].value_counts().index[0])

    else:

        trainnew[col] = trainnew[col].fillna(trainnew[col].mean())

        
#Check if any value is null

print(trainnew.isnull().any().value_counts())
dataset = trainnew.drop(['Id'], axis = 1)

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']



dataset_numeric = dataset.select_dtypes(include=numerics)
dataset_numeric.shape
nonnumeric = ['object']

dataset_nonnumeric = trainnew.select_dtypes(include=nonnumeric)
dataset_nonnumeric.shape
# Skewness in the data



dataset_numeric.skew()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

cols = dataset_numeric.columns

for c in cols:

    sns.violinplot(dataset_numeric[c])

    plt.xlabel(c)

    plt.show()

   
# Skew Correction

#log1p function applies log(1+x) to all elements of the column

skew = dataset_numeric.skew()



skewedfeatures = [s for s in skew if(s > 5.0)]

skewedfeatures

for skf in skewedfeatures:

    sk = skew[skew == skf].index[0]

    dataset_numeric[sk] = np.log1p(dataset_numeric[sk])

correlation= dataset_numeric.corr()

# Correlation tells relation between two attributes.

# Correlation requires continous data. Hence, ignore categorical data



# Calculates pearson co-efficient for all combinations

data_corr = dataset_numeric.corr()



# Set the threshold to select only highly correlated attributes

threshold = 0.5



# List of pairs along with correlation above threshold

corr_list = []



size = 36



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



dataset_numeric = dataset_numeric.drop('GarageCars', axis = 1)

dataset_numeric.shape
cols = dataset_nonnumeric.columns

split = 39

labels = []

for i in range(0,split):

    train = dataset_nonnumeric[cols[i]].unique()

    labels.append(list(set(train)))
#Import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



#One hot encode all categorical attributes

cats = []

for i in range(0, split):

    #Label encode

    label_encoder = LabelEncoder()

    label_encoder.fit(labels[i])

    feature = label_encoder.transform(dataset_nonnumeric.iloc[:,i])

    feature = feature.reshape(dataset_nonnumeric.shape[0], 1)

    #One hot encode

    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))

    feature = onehot_encoder.fit_transform(feature)

    cats.append(feature)
# Make a 2D array from a list of 1D arrays

import numpy

encoded_cats = numpy.column_stack(cats)



# Print the shape of the encoded data

print(encoded_cats.shape)
dataset_encoded = numpy.concatenate((encoded_cats,dataset_numeric.values),axis=1)
dataset_encoded.shape
#Read test dataset

dataset_test = pd.read_csv("../input/test.csv")

#Drop unnecessary columns

ID = dataset_test['Id']

dataset_test.drop('Id',axis=1,inplace=True)

dataset_test.shape
dataset_test = dataset_test.drop(removablecolumns,1)



import numpy as np



columns = dataset_test.columns

for col in columns:

    if(dataset_test[col].dtype == np.dtype('O')):

        print(5)

        print(dataset_test[col].dtype)

        print(dataset_test[col].value_counts().index[0])

        dataset_test[col] = dataset_test[col].fillna(dataset_test[col].value_counts().index[0])

        #print(trainnew[col].value_counts().index[0])

    else:

        dataset_test[col] = dataset_test[col].fillna(dataset_test[col].mean())

        #print(4)

        print(dataset_test[col].dtype)

datasettest_numeric = dataset_test.select_dtypes(include=numerics)

datasettest_nonnumeric = dataset_test.select_dtypes(include=nonnumeric)



for skf in skewedfeatures:

    sk = skew[skew == skf].index[0]

    datasettest_numeric[sk] = np.log1p(datasettest_numeric[sk])

datasettest_numeric = datasettest_numeric.drop('GarageCars', axis = 1)

datasettest_numeric.shape
#One hot encode all categorical attributes

cats = []

for i in range(0, split):

    #Label encode

    label_encoder = LabelEncoder()

    label_encoder.fit(labels[i])

    feature = label_encoder.transform(datasettest_nonnumeric.iloc[:,i])

    feature = feature.reshape(datasettest_nonnumeric.shape[0], 1)

    #One hot encode

    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))

    feature = onehot_encoder.fit_transform(feature)

    cats.append(feature)
# Make a 2D array from a list of 1D arrays

encoded_cats = numpy.column_stack(cats)
encoded_cats.shape
#Concatenate encoded attributes with continuous attributes

X_test = numpy.concatenate((encoded_cats,datasettest_numeric.values),axis=1)
X_test.shape
#get the number of rows and columns

r, c = dataset_encoded.shape



y_train = dataset_encoded[:,c-1]

X_train = dataset_encoded[:,0:c-1]
from sklearn.linear_model import Lasso

ls = Lasso(alpha = 1.0, max_iter = 100)

ls.fit(X_train, y_train)

predictions = ls.predict(X_test)


results_dataframe = pd.DataFrame({

    "Id" : ID,

    "SalePrice": predictions

})
results_dataframe.to_csv("first_submission.csv", index = False)