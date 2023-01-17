import warnings

warnings.filterwarnings('ignore')



from pandas import read_csv

dataset = read_csv("../input/pima-indians-diabetes.csv", header = None)

print(dataset.describe())
# print the first 20 rows

print(dataset.head(20))

# print dataset 

print((dataset[[1,2,3,4,5]] == 0).sum())
import numpy
# mark zero values as missing or NaN

dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)

# count the number of NaN values in each column

print(dataset.isnull().sum())
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)

# print the first 20 rows of data

print(dataset.head(20))
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



# mark zero values as missing or NaN

dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)

# split dataset into inputs and outputs

values = dataset.values

X = values[:,0:8]

y = values[:,8]

# evaluate an LDA model on the dataset using k-fold cross validation

model = LinearDiscriminantAnalysis()

kfold = KFold(n_splits=3, random_state=7)
result = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

# insert print code  

print (result.mean())
# mark zero values as missing or NaN

dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0,numpy.NaN)

# drop rows with missing values

dataset.dropna(inplace=True)

# summarize the number of rows and columns in the dataset

print(dataset.shape)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



# mark zero values as missing or NaN

dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)

# drop rows with missing values

dataset.dropna(inplace=True)

# split dataset into inputs and outputs

values = dataset.values

X = values[:,0:8]

y = values[:,8]

# evaluate an LDA model on the dataset using k-fold cross validation

model = LinearDiscriminantAnalysis()

kfold = KFold(n_splits=3, random_state=7)

result = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(result.mean())
# mark zero values as missing or NaN

dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0,numpy.NaN)
# fill missing values with mean column values

dataset.fillna(dataset.mean(), inplace=True)

# count the number of NaN values in each column

print(dataset.isnull().sum())

#
from pandas import read_csv

# use from to bring in imputer

from sklearn.impute import SimpleImputer



import numpy

# mark zero values as missing or NaN

dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0,numpy.NaN)

# fill missing values with mean column values

values = dataset.values

SimpleImputer = SimpleImputer() 

# call imputer

transformed_values = SimpleImputer.fit_transform(values)
# count the number of NaN values in each column

print(numpy.isnan(transformed_values).sum())
from sklearn.impute import SimpleImputer

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



# mark zero values as missing or NaN

dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)

# split dataset into inputs and outputs

values = dataset.values

X = values[:,0:8]

y = values[:,8]

# fill missing values with mean column values

SimpleImputer = SimpleImputer()

transformed_X = SimpleImputer.fit_transform(X)
# evaluate an LDA model on the dataset using k-fold cross validation

model = LinearDiscriminantAnalysis()

kfold = KFold(n_splits=3, random_state=7)

result = cross_val_score(model, transformed_X, y, cv=kfold, scoring='accuracy')

print(result)

#  
