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
RamanAGEsData = "/kaggle/input/AGEs.csv"
RamanInnerArmData = "/kaggle/input/innerArm.csv"
RamanThumbnailData = "/kaggle/input/thumbNail.csv"
RamanEarlobeData = "/kaggle/input/earLobe.csv"
RamanVeinData = "/kaggle/input/vein.csv"
RamanAGEsDF = pd.read_csv("../input/raman-spectroscopy-of-diabetes/AGEs.csv")
RamanInnerArmDF = pd.read_csv("../input/raman-spectroscopy-of-diabetes/innerArm.csv")
RamanThumbnailDF = pd.read_csv("../input/raman-spectroscopy-of-diabetes/thumbNail.csv")
RamanEarlobeDF = pd.read_csv("../input/raman-spectroscopy-of-diabetes/earLobe.csv")
RamanVeinDF = pd.read_csv("../input/raman-spectroscopy-of-diabetes/vein.csv")
#Let's take a quick peek at some of the Inner Arm spectral data, and make a quick plot of it: 
RamanInnerArmDF.head(25)
#Let's select the first row of the Inner Arm DataFrame to be used as the domain in future plots or analyses: 
InnerArmWavenumbersDF = RamanInnerArmDF.iloc[0]

#Let's also drop the first row (containing the Raman wavenumbers, the spectral domain) from the original DataFrame: 
RamanInnerArmDF = RamanInnerArmDF.drop(RamanInnerArmDF.index[0])
#1a) Inner Arm Data Scaling: 
from sklearn.preprocessing import StandardScaler 

#I'm making a slice of the original dataframe, from the "Var2" column onwards, as each column is an independent feature representing a wavenumber 
#chunk of the spectral domain. I'll then make a list of this dataframe of column names. This will be an input in the StandardScaler functions:
innerArmFeaturesDataFrame = RamanInnerArmDF.iloc[:,2:3159]
innerArmFeaturesList = innerArmFeaturesDataFrame.columns.tolist()

innerArmFeaturesValues = innerArmFeaturesDataFrame.loc[:,innerArmFeaturesList].values

#Separating out target values: 
innerArmTargetValues = RamanInnerArmDF.loc[:,['has_DM2']].values

#Feature Standardization: 
RamanInnerArmStandardScalerObject = StandardScaler().fit_transform(innerArmFeaturesValues)
#1b) Inner Arm PCA, set to (the reccomended) 15 components: 
from sklearn.decomposition import PCA

RamanInnerArmPCA = PCA(n_components = 15) 
RamanInnerArmPrincipalComponents = RamanInnerArmPCA.fit_transform(RamanInnerArmStandardScalerObject)

RamanInnerArmPCADF = pd.DataFrame(data = RamanInnerArmPrincipalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5', 'principal component 6', 'principal component 7', 'principal component 8', 'principal component 9', 'principal component 10', 'principal component 11', 'principal component 12', 'principal component 13', 'principal component 14', 'principal component 15'])

#Now that we have a total of fifteen new principal components, preserving as much variation from the original 3159 Raman
#wavenumber features, lets concatenate the new primary components DF with the features column in a new DataFrame:
RamanPCAInnerArmComponentsAndFeatureDF = pd.concat([RamanInnerArmPCADF, RamanInnerArmDF[['has_DM2']]], axis = 1)

#Let's get a little preview of that concatenated DataFrame: 
RamanPCAInnerArmComponentsAndFeatureDF.head(20)
#Let's drop the first and last rows of the concatenated PCA+target DataFrame, as the DM2 value
#for the first participant has, for some reason, gone "NaN": 
RamanPCAInnerArmComponentsAndFeatureDF = RamanPCAInnerArmComponentsAndFeatureDF.drop(RamanPCAInnerArmComponentsAndFeatureDF.index[0])
RamanPCAInnerArmComponentsAndFeatureDF = RamanPCAInnerArmComponentsAndFeatureDF.drop(RamanPCAInnerArmComponentsAndFeatureDF.index[19])
#2) Implementing K-Fold cross-validation and Logistic Regression, iterating one thousand times: 

#Necessary imports: 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score 

#Splitting the PCA-reduced Dataframe into Target and Feature Data: 
RamanPCAInnerArmFeaturesDF = RamanPCAInnerArmComponentsAndFeatureDF.drop(RamanPCAInnerArmComponentsAndFeatureDF[['has_DM2']], axis = 1)
RamanPCAInnerArmTargetDF = RamanPCAInnerArmComponentsAndFeatureDF.loc[:,'has_DM2']
#I'm going to drop the first rows of both Dataframes because the target value for the first partipant now registers as "NaN" for some 
#reason:
RamanPCAInnerArmFeaturesDF = RamanPCAInnerArmFeaturesDF.drop(RamanPCAInnerArmFeaturesDF.index[0])
RamanPCAInnerArmTargetDF = RamanPCAInnerArmTargetDF.drop(RamanPCAInnerArmTargetDF.index[0])

#Creating empty lists to store test accuracy results: 
avgLRScore = []
LRScoresCV5 = []

#**I keep receiving an error saying that cross_val_score cannot run correctly because there is either a NaN, infinite, or too large 
#value. I'm going to convert the Dataframes into numpy arrays to see if this is a compatibility issue**#
RamanPCAInnerArmFeaturesArray = np.asarray(RamanPCAInnerArmFeaturesDF)
RamanPCAInnerArmTargetArray = np.asarray(RamanPCAInnerArmTargetDF)

RamanPCAInnerArmComponentsAndFeatureDF.head(25)

#Outer for loop, to iterate a 1000 times. Will do this over a list over a certain integer range (1-1000)#
for i in range(1,1000):
    LRScoresCV5.append(cross_val_score(LogisticRegression(),RamanPCAInnerArmFeaturesArray,RamanPCAInnerArmTargetArray, cv = 5))

avgLRScore = np.average(LRScoresCV5)


    



avgLRScore