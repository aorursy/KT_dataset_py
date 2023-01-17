# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')

test = pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')
"""

Wanted to first look at the different data types in the dataset to 

understand which features could be used in a regression equation

Since I am building a linear regression equation, I can't use any object types (categorical)

"""



train.info()
null_columns = train.columns[train.isnull().any()]

train[null_columns].isnull().sum()
#Seeing how there is only 1 null in winPlacePerc I sorted the data to view what index the null occurs in.



train['winPlacePerc'].isnull().sort_values(ascending=False).head(10)
#Now we need to drop the values for index 2744604



train.drop([train.index[2744604]], inplace=True)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



numVar = train.drop(['winPlacePerc', 'Id', 'groupId','matchId', 'matchType'], axis=1)

scaler.fit(numVar)



scaled_feat = pd.DataFrame(scaler.transform(numVar), columns = numVar.columns)

scaled_feat.head()



#Now that we have the equivalent Z scores for all of our variables,

#I wanted to find outliers in the features and potentially remove them



boolVal = (scaled_feat > 3) | (scaled_feat < -3)



Outliers = 0



for i in scaled_feat:

    boolVal_i = boolVal[boolVal[i] == True].sum()

    Outliers = Outliers + boolVal_i

    

print(Outliers)



"""

Where there are true values, shows us how many outliers we have

as these values are more than 3 standard deviations from the mean



Returning Outliers shows us there is a massive amount of values past 3 standard deviations.

According to the empirical rules, 99.7% of our data falls within 3 standard deviations from the mean. 



With this massive amount of outliers, we can't remove them as this would significantly alter

the results of the data as we would be reducing the size of our training set by too much. 



To conclude, we will not remove outliers.

"""

col = list(train.columns)

col = col[3:] #removes the 3 categorical variables that we can't

#use in a scatterplot

col.remove('winPlacePerc') #removes our dependent variables

col.remove('matchType') #remove another categorical variable



#using a for loop, we print out jointplots for each potential

#independent variables for our regression equation on the x axis and

#dependent varaible on our y axis

for i in col:

    sns.jointplot(x=i, y='winPlacePerc', data=train)

    plt.show()
plt.figure(figsize=(18,18))

sns.heatmap(train.corr(), cmap='coolwarm', linecolor='white',

           linewidth=.5, annot=True, fmt='.2f')
#Now we will train the model for linear regression

#First step is to seperate our independent variables and dependent variables



X = train[['killPlace', 'boosts', 'walkDistance', 'weaponsAcquired']]

y = train['winPlacePerc']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3)

from sklearn.linear_model import LinearRegression

linear = LinearRegression()

linear.fit(X_train, y_train)
#Using the regression equation, we predict winPlacePerc off the 30% of data that our model hasn't seen before.

predictions=linear.predict(X_test)

c_mat = np.corrcoef(y_test, predictions)

c_xy = c_mat[0,1]

r_squared = c_xy**2

print("R Squared: {:.4f}".format(r_squared))

X_test2 = test[['killPlace', 'boosts', 'walkDistance', 'weaponsAcquired']]

test_pred = linear.predict(X_test2)



results = pd.DataFrame(index=test['Id'], data=test_pred)

results.rename(columns={0: 'winPlacePerc'}, inplace=True)



results.to_csv('submission.csv')