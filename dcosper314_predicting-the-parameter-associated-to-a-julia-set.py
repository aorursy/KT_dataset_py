import numpy as np 

import pandas as pd

import cv2

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from sklearn.feature_selection import VarianceThreshold

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error
train_df = pd.read_csv("../input/julia-sets/Julia_training_df.csv",index_col=0)

test_df = pd.read_csv("../input/julia-sets/Julia_test_df.csv",index_col=0)
print(train_df.shape)

print(test_df.shape)
print(train_df.iloc[:,-2:].head())

print(test_df.iloc[:,-2:].head())
image_1 = cv2.imread('../input/julia-set-pics/julia_-0.1484750832142412_0.49057950471753653.pgm')

image_2 = cv2.imread('../input/julia-set-pics/julia_-0.32511910466582794_0.18720289197856887.pgm')

image_3 = cv2.imread('../input/julia-set-pics/julia_-0.9402198478635815_-1.4458258934979922.pgm')

#Note the file description contains the complex parameter a + ib associated to the Julia set





#fig, axes = plt.subplots(2,2,subplot_kw=dict(polar=True))

#axes[0,0].imshow(image_1)

#axes[0,1].imshow(image_2)

#axes[1,0].imshow(image_3)

#plt.show



plt.imshow(image_1)

plt.show()

plt.imshow(image_2)

plt.show()

plt.imshow(image_3)

plt.show()

train_df.hist(column = ['0','1','19900','20100'],bins = 10)
X1 = train_df.iloc[:,0:-2]

y1 = train_df.iloc[:,-2:]



X2 = test_df.iloc[:,0:-2]

y2 = test_df.iloc[:,-2:]

# Separating dataframes into dependent and independent features.



selector = VarianceThreshold(15.0)

# We may come back and take a closer look at the choice of 15 here if we want to tweak some aspects of our model.



selector.fit(X1)

mask = selector.get_support()

# mask is a boolean array of which columns pass and fail the variance threshold.



X1 = X1.loc[:,mask]

X2 = X2.loc[:,mask]

# We obtain the mask from X1 and apply it to both X1 and X2 to keep our features consistent



print(X1.shape)

print(X2.shape)

regressor = DecisionTreeRegressor(max_leaf_nodes = 15, random_state=0)

regressor.fit(X1,y1)



y2_predict = pd.DataFrame(regressor.predict(X2))

print(mean_absolute_error(y2, y2_predict))
y2.head()
y2_predict.head()
# Plot the results

plt.figure()

s = 25

#plt.scatter(y1.iloc[:, 0], y1.iloc[:, 1], c="navy", s=s,edgecolor="black", label="Data")

plt.scatter(y2.iloc[:, 0], y2.iloc[:, 1], c="green", s=s,edgecolor="black", label="Test")

plt.scatter(y2_predict.iloc[:, 0], y2_predict.iloc[:, 1], c="red", s=s, edgecolor="black", label="Prediction")

plt.xlim([-2, 2])

plt.ylim([-2, 2])

plt.xlabel("RE")

plt.ylabel("IM")

plt.title("Parameters of Julia Sets")

plt.legend(loc="best")

plt.show()
X1 = train_df.iloc[:,0:-2]

y1 = train_df.iloc[:,-2:]



X2 = test_df.iloc[:,0:-2]

y2 = test_df.iloc[:,-2:]

#Redefining my data sets since I altered them in the previous experiment.



X1 = X1.assign(Mean=X1.mean(axis=1))

X2 = X2.assign(Mean=X2.mean(axis=1))

#Now we assign the mean as a column
X1.hist(column = ['Mean'],bins = 15)

#Note that 250 is white while 0 is black.
indices = X1.index[X1['Mean'] <= 235].tolist()

plt.scatter(y1.iloc[:, 0], y1.iloc[:, 1], c="navy", s=s,edgecolor="black", label="Data")

for i in indices:

    plt.scatter(y1.iloc[i, 0], y1.iloc[i, 1], c="red", s=s,edgecolor="black", label="Test")





plt.xlim([-2, 2])

plt.ylim([-2, 2])

plt.show()