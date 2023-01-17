import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt # data visualisation

import seaborn as sns  # data visualisation

%matplotlib inline
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#importing combined data: 

# cbb.csv



CBB = pd.read_csv("../input/college-basketball-dataset/cbb.csv")
CBB.head()
CBB.info()
CBB.shape
#Checking missing value 



def missing_check(CBB):

    total = CBB.isnull().sum().sort_values(ascending=False)   # total number of null values

    percent = (CBB.isnull().sum()/CBB.isnull().count()).sort_values(ascending=False)  # percentage of values that are null

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])  # putting the above two together

    return missing_data # return the dataframe

missing_check(CBB)
CBB["YEAR"] = pd.Categorical(CBB["YEAR"])
CBB.describe()
#checking for skewness in a data

CBB.skew()
CBB.columns
#cheking for outliers in a data through boxplot

plt.figure(figsize= (25,10))

plt.subplot(19,1,1)

sns.boxplot(x=CBB.G , color='blue')



plt.subplot(19,1,2)

sns.boxplot(x= CBB.W, color='red')



plt.subplot(19,1,3)

sns.boxplot(x= CBB.ADJOE, color='green')



plt.subplot(19,1,4)

sns.boxplot(x=CBB.EFG_O , color='blue')



plt.subplot(19,1,5)

sns.boxplot(x= CBB.ADJDE, color='red')



plt.subplot(19,1,6)

sns.boxplot(x= CBB.BARTHAG, color='green')



plt.subplot(19,1,7)

sns.boxplot(x=CBB.EFG_D , color='blue')



plt.subplot(19,1,8)

sns.boxplot(x= CBB.TOR, color='red')



plt.subplot(19,1,9)

sns.boxplot(x= CBB.TORD, color='green')



plt.subplot(19,1,10)

sns.boxplot(x= CBB.ORB, color='red')



plt.subplot(19,1,11)

sns.boxplot(x= CBB.DRB, color='green')



plt.subplot(19,1,12)

sns.boxplot(x=CBB.FTR , color='blue')



plt.subplot(19,1,13)

sns.boxplot(x= CBB.FTRD, color='red')



plt.subplot(19,1,14)

sns.boxplot(x= CBB['2P_O'], color='green')



plt.subplot(19,1,15)

sns.boxplot(x=CBB['2P_D'], color='blue')



plt.subplot(19,1,16)

sns.boxplot(x= CBB['3P_O'], color='red')



plt.subplot(19,1,17)

sns.boxplot(x= CBB['3P_D'], color='green')



plt.subplot(19,1,18)

sns.boxplot(x=CBB.ADJ_T , color='blue')



plt.subplot(19,1,19)

sns.boxplot(x= CBB.WAB, color='red')



plt.show()
CBB_Outlier_Treatment = CBB.drop(columns = ["TEAM", "CONF", "POSTSEASON","SEED","YEAR"])

CBB_Outlier_Treatment
from scipy import stats

z = np.abs(stats.zscore(CBB_Outlier_Treatment))   # get the z-score of every value with respect to their columns

print(z)
threshold = 3 # In a Normal distribution standard deviation is within or equal to 3 times

print ("Rows and columns location showing outlier value:")

np.where(z > threshold)
print(z[0][0]) # for example
CBB_copy = CBB_Outlier_Treatment.copy()   #make a deep copy of the dataframe



#Replace all the outliers with median values. This will create new some outliers but, we will ignore them



for i, j in zip(np.where(z > threshold)[0], np.where(z > threshold)[1]):# iterate using 2 variables.i for rows and j for columns

    CBB_copy.iloc[i,j] = CBB_Outlier_Treatment.iloc[:,j].median()  # replace i,jth element with the median of j i.e, corresponding column
z = np.abs(stats.zscore(CBB_copy))

np.where(z > threshold)  # New outliers detected after imputing the original outliers

sns.distplot(CBB_Outlier_Treatment.G);
sns.distplot(CBB_Outlier_Treatment.W);
sns.distplot(CBB_Outlier_Treatment.ADJOE);
sns.pairplot(CBB_Outlier_Treatment, kind= "reg"); 
CBB_Outlier_Treatment.corr() # Method = Pearson
plt.figure(figsize= (30,20))

sns.heatmap(CBB_Outlier_Treatment.corr(), annot = True);
pd.crosstab([CBB.TEAM,CBB.CONF,CBB.YEAR], CBB['W']).head(10)
pd.crosstab([CBB.TEAM,CBB.CONF,CBB.YEAR], CBB['W']).tail(10)

CBB_dummies= pd.get_dummies(CBB, prefix='year', columns=['YEAR']) #This function does One-Hot-Encoding on categorical text
CBB_dummies.head()
CBB_dummies.corr() # now we can analyze the relationship between variable year wise
plt.figure(figsize= (30,20))

sns.heatmap(CBB_dummies.corr(), annot = True);