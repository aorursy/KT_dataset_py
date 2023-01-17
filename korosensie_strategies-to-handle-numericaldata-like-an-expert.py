#Problem---Rescale the numerical values of the features

#Solution---We will use MinMaxScaler to rescale



#importing numpy and sklearn library

import numpy as np

from sklearn import preprocessing



#creating features

feature = np.array([[-500.5],

                   [-100.1],

                   [0],

                   [100.1],

                   [900.9]])



#creating scaler

minmax_scale=preprocessing.MinMaxScaler(feature_range=(0,1))

#scaling feature

scaled_feature=minmax_scale.fit_transform(feature)
#displaying rescaled feature

scaled_feature
#Problem---rescale the features with mean of 0 and standard deviation of 1

#Solution---We will use sklearns standardScaler



#importing numpy and sklearn library

import numpy as np

from sklearn import preprocessing



#creating features

feature = np.array([[-1000.5],

                   [-200.1],

                   [500],

                   [600.1],

                   [9900.9]])



#creating scaler

scaler= preprocessing.StandardScaler()



#Transforming feature

standardized = scaler.fit_transform(feature)
#displaying feature after being standardized

standardized
#To check the mean and standard deviation of the feature

print("Mean--",round(standardized.mean()))

print("Standard deviation---",standardized.std())
#importing numpy and sklearn library

import numpy as np

from sklearn import preprocessing



#creating features

feature = np.array([[-1000.5],

                   [-200.1],

                   [500],

                   [600.1],

                   [9900.9]])



#creating scaler

scaler= preprocessing.RobustScaler()



#Transforming feature

standardized = scaler.fit_transform(feature)
standardized
#Problem---Rescale the values of observation to unit norm(length is 1)

#Solution---We will use sklearn's Normalizer to rescale in unit norm



#importing libraries

import numpy as np

from sklearn.preprocessing import Normalizer



#creating a feature Matrix

features =np.array([[0.5,0.5],

                    [1.1, 3.4],

                    [1.5, 20.2],

                    [1.63, 34.4],

                    [10.9, 3.3]])



#creating Normalizer using norm Eculidean form

normalizer =Normalizer(norm='l2')



#transforming the feature matrix

normalizer.transform(features)
##creating Normalizer using norm Manhattan form

normalizer =Normalizer(norm='l1')



#transforming the feature matrix

normalizer.transform(features)
##creating Normalizer using norm max form

normalizer =Normalizer(norm='max')



#transforming the feature matrix

features_l1_norm=normalizer.transform(features)



features_l1_norm
#Problem---Create a polynomial and interaction feature

#Solution---We will use sklearn PolynomialFeatures



#importing sklearn and numpy libraries

import numpy as np

from sklearn.preprocessing import PolynomialFeatures



#creating a feature matrix

features=np.array([[2,3],

                  [1,2],

                  [3,4]])



#Creating a PolynomialFeatures object

polynomial_interaction = PolynomialFeatures(degree =2,include_bias=False)



#creating polynomial features

polynomial_interaction.fit_transform(features)
#We can restrict features created to only interaction features 



interaction=PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)



interaction.fit_transform(features)
#Problem---Transform One or more features

#Solution---We will use FunctionTransformer and apply



#importing numpy and sklearn

import numpy as np

from sklearn.preprocessing import FunctionTransformer



#creating a features matrix

features = np.array([[2, 3],

                     [2, 3],

                     [2, 3]])

#creating a function to add 10

def add_10(x):

    return x+10



#creating FunctionTransformer object

ten_transformer =FunctionTransformer(add_10)



#Transforming the feature matrix

ten_transformer.transform(features)
#using pandas apply 



#importing pandas library

import pandas as pd



#creating DataFrame

df=pd.DataFrame(features,columns=['features1','features2'])



#applying function

df.apply(add_10)
#Problem---Find the outliers of the observation

#Solution---We will use sklearn to find the outliers



#importing sklearn and numpy library

import numpy as np

from sklearn.covariance import EllipticEnvelope

from sklearn.datasets import make_blobs



#creating simulated data

features,_=make_blobs(n_samples=10,

                     n_features =2,

                     centers=1,

                     random_state=1)



#replacing the first values of the observation with outlier value

features[0,0]=10000

features[0,1]=10000





#creating outlier detector

outlier_detector =EllipticEnvelope(contamination=.1)



#fit detector

outlier_detector.fit(features)



#predicting outliers

outlier_detector.predict(features)



#the -1 will show that it is outlier
#Problem---Handle the outliers

#Solutions---We will see 3 ways to handle the outliers first is to drop them



#importing the library

import pandas as pd



#creating a dataframe

houses =pd.DataFrame()

houses['Price'] = [534433, 392333, 293222, 4322032]

houses['Bathrooms'] = [2, 3.5, 2, 116]

houses['Square_Feet'] = [1500, 2500, 1500, 48000]





#Filtering the obseravtions

houses[houses['Bathrooms']<20]
#Secondly,we can mark them as outliers and include it as a feature:



#importing library

import numpy as np



#creating features based on boolean condition

houses['Outlier']=np.where(houses['Bathrooms']<20,0,1)



#show data

houses
#Finally we can transform the feature to reduce or dampen the effect of the outlier



#log feature

houses['Log_of_square_feet']=[np.log(x) for x in houses['Square_Feet']]



#displaying data

houses
#Problem---Break the numerical features into discrete bins

#Solution---There are two techniques to break the data.First to binarize the feature according to some thresold



#importing numpy and sklearn library

import numpy as np

from sklearn.preprocessing import Binarizer



#creating feature

age=np.array([[6],

              [12],

              [20],

              [36],

              [65]])



#Creating Binarizer

binarizer =Binarizer(18)



#transforming the feature

binarizer.fit_transform(age)
#or we can use np digitize



#bin feature

np.digitize(age,bins=[18])
#Second, we can break up numerical features according to multiple thresholds



#Bin Feature

np.digitize(age,bins=[20,30,64])
#The bin 20 mean greater than 20 without including 20 We can include 20 by adding right=True Parameter

np.digitize(age,bins=[20,30,64],right=True)
#Problem---Handle the missing values from the data

#Solution---We will use Numpy to delete missing values



#importing numpy library

import numpy as np



#creating feature

features=np.array([[1.1, 11.1],

                   [2.2, 22.2],

                   [3.3, 33.3],

                   [4.4, 44.4],

                   [np.nan, 55]])



#Keeping only those observation that are not missing (denoted by ~)

features[~np.isnan(features).any(axis=1)]
#We can drop missing observations



#imorting pandas library

import pandas as pd



#loading dataframe

dataframe=pd.DataFrame(features,columns=['feature1','feature2'])



#Deleting observation with missing values

dataframe.dropna()
#Problem---Fill the missing values of the obseravtion

#Solution---We will use sklearn Imputer to fill missing values



#importing numpy library

import numpy as np

#importing library

from sklearn.impute import SimpleImputer





#creating feature

features=np.array([[1.1, 11.1],

                   [2.2, 22.2],

                   [3.3, 33.3],

                   [4.4, 44.4],

                   [np.nan, 55]])



#Creating Imputer object

 

mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')





#imputing values

features_mean_imputed =mean_imputer.fit_transform(features)



features_mean_imputed