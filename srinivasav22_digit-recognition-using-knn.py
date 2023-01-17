import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np    # For array operations

import pandas as pd   # For DataFrames
train = pd.read_csv('/kaggle/input/mnist-digit-recognition-using-knn/train_sample.csv')

test = pd.read_csv('/kaggle/input/mnist-digit-recognition-using-knn/test_sample.csv')
train.head()
X_train = train.iloc[:,1:]

y_train = train.iloc[:,0]



print(X_train.shape)

print(y_train.shape)
X_test = test.iloc[:,1:]

y_test = test.iloc[:,0]



print(X_test.shape)

print(y_test.shape)
s = np.random.choice(range(X_train.shape[0]), size=12)

list(enumerate(s))
import matplotlib.pyplot as plt    # For plotting 

%matplotlib inline                 



plt.figure(figsize=(10,5)) #Width and heigth of the image displayed below (Optional)



for i,j in enumerate(s):

    plt.subplot(4,6,i+1)                                      # Subplot flag

    plt.imshow(np.array(X_train.loc[j]).reshape(28,28))      # Plot the image

    plt.title('Digit: '+str(y_train.loc[j]))                 # Target of the image

    plt.gray()                                                # For gray scale images 
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(algorithm='brute', n_neighbors=5, p=2, weights='distance')

# 'brute' for searching through all the samples

# p=2 for Euclidean; p=1 for Mannhatten Distance

# Check the help file for all the arguments



knn.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix



pred_train = knn.predict(X_train) 

cm_test = confusion_matrix(y_true=y_train, y_pred=pred_train)

print(cm_test)
#pd.crosstab(y_train, pred_train, rownames=['True'], colnames=['Predicted'])
pred_test = knn.predict(X_test) 

cm_test = confusion_matrix(y_true=y_test, y_pred=pred_test)

print(cm_test)
#pd.crosstab(y_test, pred_test, rownames=['True'], colnames=['Predicted'])
acc_test = float(np.trace(cm_test))/np.sum(cm_test)

print(acc_test)
#r = 0

r = np.random.choice(X_test.shape[0],1)    # Uncomment this to set a random query

print(r)



query = X_test.iloc[r].values

#print(query)
nn=5                        # Number of search results

out = knn.kneighbors(n_neighbors=nn, return_distance=True, X=query)  # Print 'out' and Check the object type

print(out)

#print(type(out))

#out(knn.kneighbors) is a tuple. You will get an error if you run this cell for 2nd time
distances = out[0]          # Distance of each retrieved sample

results = out[1][0]            # Retrieved sample index

print(distances)         

print(results)
# Plot Qeury Vs Search Results

plt.figure(figsize=(15,5))

plt.subplot(1,nn+1,1)

plt.imshow(np.array(query).reshape(28,28))

plt.title('Query: '+str(y_test.iloc[r]))

plt.gray()



for i,j in enumerate(results):

    plt.subplot(1,nn+1,i+2)

    plt.imshow(np.array(X_train.iloc[j]).reshape(28,28))

    plt.title('Neighbor '+str(i+1)+': '+str(y_train.iloc[j]))

    plt.gray()
## Randomly Generate some Data

import numpy as np

import pandas as pd

from sklearn import preprocessing
#Generate 1000 rows and 4 columns randomly -- size=(1000,4)

#Each value ranges between 2 and 100

#The column names are T, A, B, C (T is target)



data  = pd.DataFrame(np.random.randint(2,100,size=(10000, 4)), columns=list('TABC'))
#Activity1: #Find the Dimesnions of Dataframe



#Activity2: #Display first 5 rows



#Activity3: #Display last 10 rows



#data
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2) #80-20 ratio train test split

print(train.shape, test.shape)
#Store train target as Y_train

Y_train = train["T"]
#Store test target as y_test

y_test = test["T"]
#Normalize the data (x-min(x))/(max(x)-min(x))

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))



#Standardize the data (x-mean(x))/std(x)

#scaler = preprocessing.StandardScaler()



scaler.fit(train.iloc[:,1:])



stdtrain = pd.DataFrame(scaler.transform(train.iloc[:,1:]), columns=list("abc"))

stdtest = pd.DataFrame(scaler.transform(test.iloc[:,1:]), columns=list("abc"))
print(stdtrain.head(5))

print(stdtest.head(5))
#X_train = stdtrain.iloc[:,1:]

#y_train = stdtrain.iloc[:,0]

print(stdtrain.shape)

print(Y_train.shape)
stdtrain.head(5)
#X_test = stdtest.iloc[:,1:]

#y_test = stdtest.iloc[:,0]



print(stdtest.shape)

print(y_test.shape)
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=5, p=2)

knn.fit(stdtrain, Y_train)
predictions = knn.predict(stdtest)
#Activity 4: Function to calculate mse

#mse(predictions,y_test)
#to call inbuilt error metrics from sklearn

from sklearn import metrics
metrics.mean_squared_error(predictions, y_test)