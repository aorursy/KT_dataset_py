# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

#Installing seaborn

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split



#read file

phone = pd.read_csv("../input/smartphone.csv", index_col=0)



phone.head(5)
# Function definitions

def EucDist(x1, y1, x2, y2):

    return np.sqrt( np.power((x1-x2),2) + np.power((y1-y2), 2) )
#looking at structure of data

#phone.dtypes

#Removing -1 position ID

df = phone.drop(phone.loc[phone['posId']<=0].index, inplace=True)





#Dropping index

df = phone.drop(phone.columns[[0]], axis=1)
#examining the min and max of the data



df.describe()




#heatmap



ax = sns.heatmap(df, cmap="YlGnBu")
#plotting correlation in a heat matrix 

plt.figure(figsize=(15,5))

sns.heatmap(df.corr(),annot=True, cmap='coolwarm', linewidth=0.5)
X = df['x']

#print(X)

Y = df['y']

#lons



Train1, Test1, xTrain, xTest = train_test_split(df, X, test_size=0.3, random_state=2)

Train2, Test2, yTrain, yTest = train_test_split(df, Y, test_size=0.3, random_state=2)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.decomposition import PCA
#xTrain = StandardScaler().fit_transform(xTrain)

#pca = PCA(n_components=3)

#principalComponents = pca.fit_transform(xTrain)

#principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])

#print (principalDf)

#pca = PCA(n_components=3)

#fit = pca.fit(xTrain)

#print(fit.components_)

#print(fit.explained_variance_ratio_)

#print(fit.explained_variance_ratio_)

#print('explained variance ratio (first three components): %s'

    #  % str(fit.explained_variance_ratio_))

from sklearn.ensemble import ExtraTreesClassifier

#Tree Classifier for variable selection 

#model = ExtraTreesClassifier()

#model.fit(X, Y)
from sklearn.ensemble import ExtraTreesClassifier

# load data

#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

#names = ['MagneticFieldX', 'MagneticFieldY', 'MagneticFieldZ', 'Z.AxisAgle.Azimuth.', 'X.AxisAngle.Pitch.', 'Y.AxisAngle.Roll.', 'posId', 'x', 'y']

#print(names)

#dataframe = (phone1,names)

#print(dataframe)

#dataframe.describe

#array = dataframe.values

#X = array[:,0:8]

#Y = array[:,8]

# feature extraction

#model = ExtraTreesClassifier()

#model.fit(X, Y)

#print(model.feature_importances_)

#df=phone1

#print(df)
print(df)
rfX = RandomForestRegressor(n_estimators=500,

                              min_samples_leaf=20,

                              max_features="sqrt",

                              oob_score=True,

                              n_jobs=-1,

                              verbose=1)

rfX.fit(Train1, xTrain)

predX = rfX.predict(Test1)

print(predX)

#print(latTest)



#errors = abs(predX - Test1)

#print(errors)

# Print out the mean absolute error (mae)

#print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
rfY = RandomForestRegressor(n_estimators=500,

                              min_samples_leaf=20,

                              max_features="sqrt",

                              oob_score=True,

                              n_jobs=-1,

                              verbose=1)

rfY.fit(Train2, yTrain)

predY= rfY.predict(Test2)

print(predY)
dists = EucDist(predX, predY, xTest, yTest)

meanED = np.mean(dists)

maxED = np.max(dists)

minED = np.min(dists)

print("--Metrics for Random Forests for Smartphone--")

print("meanED = " + str(meanED) + " m")

print("maxED = " + str(maxED) + " m")

print("minED = " + str(minED) + " m")