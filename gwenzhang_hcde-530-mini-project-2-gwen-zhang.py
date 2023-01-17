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
#Create a DataFrame with the csv file.
df = pd.read_csv('../input/airbnb-new-york-city-listing-info/listings.csv', index_col=0, low_memory=False)
#select the necessary columns and drop the row contains NaN
select = df[['number_of_reviews','amenities']]
#drop all rows that contains NaN.
select = select.dropna(axis=0)
#drop all rows that review number is 0.
select = select[(select != 0).all(1)]

#Encodingd the data:
#re-format the amenities column to a list of columns which use 0 or 1 to represent whether each record contains this amenity. 
from sklearn import preprocessing
import numpy as np

#Extra a full list of features from amenities column
featuresSet = {'Internet','Wifi'}
for amenities in df["amenities"]:
    options = amenities[1:-1].split(",")
    for option in options:
        featuresSet.add(option)

featureList = list(featuresSet)
print(featureList)
len(featureList)
#grouping the review number into three bucket by frequency distribution
#categorize the decision making level into three buckets baesd on review number
# bucket1 --> "basic"
# bucket2 --> "moderate"
# bucket3 --> "attractive"
from collections import OrderedDict  
import matplotlib.pyplot as plt 
  
def group_list(inputList):  
    result =  [(el, inputList.count(el)) for el in inputList] 
    return list(OrderedDict(result).items()) 

array = []
for a, b in select.iterrows():
    array.append(b[0])
    
reviewDistribution = group_list(array)

#review number distribution plot
plt.plot(reviewDistribution)
reviewDistribution.sort()
#auto generate the right bucketing for result.
# But it doesn't work properly, manually adjust
# the result bucketing in the end.
def generateBucketing(reviewDistribution):
    
    totalCount = 0
    for x in reviewDistribution:
        totalCount += x[1]
    step = int(totalCount * 0.01)
    step2 = step
    #scan from front to back and from back to
    #front at the same time, both travel totalCount / 3 steps
    pFront = 0
    pEnd = len(reviewDistribution) - 1
    while step > 0: 
        step -= reviewDistribution[pFront][1]      
        pFront += 1
        
    while step2 > 0:
        step2 -= reviewDistribution[pEnd][1]
        pEnd -= 1
        
    return (pFront, pEnd)

bucket = generateBucketing(reviewDistribution)
print("Auto bucketing of Y looks like below:")
print("basic bucket: " + "[1," + str(bucket[0]) + "]")
print("moderate bucket: " + "[" +  str(bucket[0] + 1) + "," + str(bucket[1]) + "]")
print("attractive bucket: " + "[" +  str(bucket[1] + 1) + "," + str(reviewDistribution[-1][0]) + "]" + "\n")

# Manually adjust and make sense of the buckeing to [1,12], [13,48],[49,746]
print("Adjusted bucketing of Y looks like below:")
print("basic bucket: " + "[1, 12]")
print("moderate bucket: " + "[13, 48]")
print("attractive bucket: " + "[49, 746]")
#Encode and re-format the data set.

#transformX function will transform the value of amenities column to a list of 0 and 1, 
#which represent whether contains the amenity in featureList, and it's in order. 
def transformX(featureList, dfRow):
    #list comprehension, create an length 150 array contains only int 0.
    encodedList = [0 for range in range(0,150)]
    #scan the dfRow, flip the index of existed amenity option in feature list to 1.
    options = dfRow[1:-1].split(",")
    for option in options:
        encodedList[featureList.index(option)] = 1
    return encodedList

#transformY function will bucketize Y based on below mapping. 
# Basic bucket: [1, 12]
# Moderate bucket: [13, 48]
# Attractive bucket: [49, 746]
def transformY(y):
    if y <= 12:
        return 0
    elif y <= 48:
        return 1
    return 2

def encode(select):
    x_all = []
    y_all = []
    #encoding the amenities column
    for index, value in select.iterrows():
        x_all.append(transformX(featureList, value[1]))
        y_all.append(transformY(value[0]))
    return (x_all, y_all)

X_all, Y_all = encode(select)
from sklearn.model_selection import train_test_split

#Split trainig set and testing set. 20% test; 80% training
num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, Y_all, test_size=num_test, random_state=23)
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
coeff = pd.DataFrame(regr.coef_, featureList, columns=['Coefficient'])  
coeff
#The larger the coefficient, the more impactful this amenity is.
#The smaller the coefficient, the more essential this amenity is.
coeff = coeff.sort_values(by=['Coefficient'])
coeff.head(5)
coeff.tail(10)
