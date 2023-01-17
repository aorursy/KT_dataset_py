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
df = pd.read_csv('/kaggle/input/wine-quality/winequalityN.csv')

df.head()

df1 = df.fillna(df.mean())
# Convert string to number: 

def transform_type(val):

    if val == 'white':

        return 0

    else:

        return 1

    
# apply to 'type' column

df1['type'] = df1['type'].apply(transform_type)
df1.head()
# Choose X for input, y for output



input_data = X = df1[['type', 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']].values

target = y = df1['quality']

# Split data with 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print('Train set:', X_train.shape, y_train.shape)

print('Test set:', X_test.shape, y_test.shape)
# Set up KNN model

from sklearn.neighbors import KNeighborsClassifier



#import metrics



from sklearn import metrics
Ks = 10

# Mean accuracy:  This function is equal to the jaccard_similarity_score function

mean_acc = np.zeros((Ks-1))

# Compute the standard deviation along the specified axis.



std_acc = np.zeros((Ks-1))



for n in range(1, Ks):

    

    #Train model and Predict

        neigh =  KNeighborsClassifier(n_neighbors = n).fit(X_train, y_train)

        yhat = neigh.predict((X_test))

        mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

        

        std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

        

mean_acc

        
print(mean_acc.max(), "with k = ", mean_acc.argmax()+1)
import matplotlib.pyplot as plt

plt.plot(range(1,Ks),mean_acc,'g')

plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.1)

plt.legend(('Accuracy ', '+/- 3xstd'))

plt.ylabel('Accuracy ')

plt.xlabel('Number of Nabors (K)')

plt.tight_layout()

plt.show()