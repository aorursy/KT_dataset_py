# Important libraries

import os # Read files in environment

import numpy as np # Linear algebra

import pandas as pd # Data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb # Plotting

import matplotlib.pyplot as plt # Plotting

import sklearn # ML 

from sklearn import preprocessing # Scaling and centering

from sklearn.model_selection import train_test_split # Splitting dataset

from sklearn.decomposition import PCA # PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # LDA
# Obtaining files

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Making dataframes

train = pd.read_csv("../input/hdfc-hiring-september-2019/Train.csv")

test = pd.read_csv("../input/hdfc-hiring-september-2019/Test.csv")



# Saving ID column

MyColumn = test["Col1"]
# Avoiding scientific notation

pd.set_option("display.float_format", lambda x: "%.3f"%x)
# Checking data

train.head(10)
# Eliminating some columns with missing values

train = train.loc[:, train.isnull().sum() < 0.1*train.shape[0]]

col = list(train.columns[2:])

col = ["Col1"] + col

test = test[col]

test.insert(loc = 1, column = "Col2", value = ["" for i in range(test.shape[0])])
# Filling in columns with missing values

def Fill(data):

    for column in data.columns[2:]:

        med = np.nanmedian(data[column].values)

        data[column].fillna(med, inplace = True) 

    return data



train = Fill(train)

test = Fill(test)
# Scaling and centering

# Mean = 0

# Pop. std. dev. = 1

# Sum of sq. = No. of rows

def ScaleCenter(data):

    data.iloc[:, 2:] = preprocessing.scale(data.iloc[:, 2:])

    return data    



train = ScaleCenter(train)

test = ScaleCenter(test)
# Finding optimal number of principal components

X = train.iloc[:, 2:]

pca = PCA() 

pca.fit(X) 

var = pca.explained_variance_ratio_

CumSum = np.cumsum(var) 

num_pc = np.argmax(CumSum >= 0.99) + 1

# Finding principal components on train data

pca = PCA(n_components = num_pc)

PC_train = pca.fit_transform(X)

# Using the same coefficients for test data

PC_test = pca.transform(test.iloc[:, 2:]) 



# Make new DataFrame with apropriate column names

columns = []

for i in range(num_pc):

    columns.append("PC" + str(i+1))

    

def PCDF(data, PC, columns):

    df = pd.DataFrame(data = PC, columns = columns)

    pcdf = pd.concat([data[["Col2"]], df], axis = 1)

    return pcdf



train = PCDF(train, PC_train, columns)

test = PCDF(test, PC_test, columns)
# Scree plot

var = var*100

var_data = var[0:num_pc]

plt.bar(x = range(num_pc), height = var_data)

plt.xticks(rotation = "vertical")

plt.ylim(0, var_data[0]+1)

plt.show()
# Checking for class balance

class_data = train.Col2.value_counts()

class_data.plot(kind = "bar", color = ["red", "blue"])

print(class_data)
# Splitting train data

train_data, test_data, train_labels, test_labels = train_test_split(train[train.columns[1:]], train["Col2"], test_size = 0.25)
# Modeling

classifier = LinearDiscriminantAnalysis()

classifier.fit(train_data, train_labels)

prediction = classifier.predict(test_data)

confusion_matrix = sklearn.metrics.confusion_matrix(test_labels, prediction)

accuracy = sklearn.metrics.accuracy_score(test_labels, prediction)

f1 = sklearn.metrics.f1_score(test_labels, prediction)
# Using model for prediction

classifier.fit(train.iloc[:, 1:], train["Col2"])

prediction = classifier.predict(test.iloc[:, 1:])
# Saving predictions

data = pd.DataFrame({ 

    "Col1" : list(MyColumn.values),

    "Col2" : list(prediction)

})

data.to_csv("submit.csv", index = False) 