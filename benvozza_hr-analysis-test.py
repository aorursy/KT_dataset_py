# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler



from sklearn.linear_model import LinearRegression

%matplotlib inline



# Input data files are available in the "../input/" directory.



hrData = pd.read_csv('../input/HR_comma_sep.csv')

hrData = hrData.drop(['sales'], axis=1)

hrData['salary'].replace(['low', 'medium', 'high'], [0, 1, 2], inplace = True)

# Sales column doesn't make sense, doesn't match column description (data being clear is important)

hrData.head()
#Look at how which section has the largest variance (varies the most among employees)

largestVar = 0

largestVarCol = ""

for i in list(hrData):

    x = np.var(hrData[i])

    # Normalise Variance

    x = x / len(hrData[i])

    print(x)

    if x > largestVar:

        largestVar = x

        largestVarCol = i

print(largestVarCol)
# Looking at the linear correlation between features will demonstrate how 

# satisfaction levels can be predicted from the remaining information

hrData.corr()
# split data 75%-25% fror training and testing (15000 total values)



trainingDataY = hrData['satisfaction_level'][:12000]

trainingDataX = hrData.drop(['satisfaction_level'], axis=1)[:12000]

testDataY = hrData['satisfaction_level'][12000:]

testDataX = hrData.drop(['satisfaction_level'], axis=1)[12000:]





lm = LinearRegression()

lm.fit(trainingDataX , trainingDataY)

allFeatScore = lm.score(testDataX, testDataY)



from sklearn.linear_model import Ridge



lr = LinearRegression()

lr.fit(hrData[['last_evaluation', 'left', 'number_project' ]][:12000], trainingDataY)

threeFeatScore = lr.score(hrData[['last_evaluation', 'left', 'number_project' ]][12000:], testDataY)

print("all feature score = ", allFeatScore)

print("three feature score = ", threeFeatScore)

lr = Ridge(alpha=0.5)

lr.fit(hrData[['last_evaluation', 'left', 'number_project' ]][:12000], trainingDataY)

threeFeatScore = lr.score(hrData[['last_evaluation', 'left', 'number_project' ]][12000:], testDataY)

print("three feature score with ridge = ", threeFeatScore)


xData = StandardScaler().fit_transform(hrData[['last_evaluation', 'left', 'number_project' ]])

lr = LinearRegression()

lr.fit(xData[:12000], trainingDataY)

threeFeatScore = lr.score(xData[12000:], testDataY)

print("three feature score with standardised data = ", threeFeatScore)
# K chosen randomly, If time allowed would have added Elbow Method for K Choice



from sklearn.cluster import KMeans

K = 5

km = KMeans(n_clusters = K)

km.fit(hrData)

clusterLabels = km.labels_



# View the results

# Set the size of the plot

plt.figure(figsize=(14,7));

 

# Create a colormap

colormap = np.array(['red', 'lime', 'black', 'blue', 'yellow'])

 

    

# Plot the Models Classifications

plt.subplot(1, 2, 2);

plt.scatter(hrData["satisfaction_level"], hrData["last_evaluation"], c=colormap[clusterLabels], s=40);

plt.title('K Mean Classification');



plt.figure(figsize=(14,7));

 

# Create a colormap

colormap = np.array(['red', 'lime', 'black', 'blue', 'yellow'])



plt.scatter(hrData["satisfaction_level"], hrData["left"], c=colormap[clusterLabels], s=40);

plt.title('K Mean Classification 2');



plt.figure(figsize=(14,7));

 

# Create a colormap

colormap = np.array(['red', 'lime', 'black', 'blue', 'yellow'])



plt.scatter(hrData["number_project"], hrData["satisfaction_level"], c=colormap[clusterLabels], s=40);

plt.title('K Mean Classification 3');
from sklearn.ensemble import RandomForestRegressor



# Apply Random Forest



rf = RandomForestRegressor()

rf.fit(hrData[['last_evaluation', 'left', 'number_project' ]][:12000], trainingDataY)

rf1score = rf.score(hrData[['last_evaluation', 'left', 'number_project' ]][12000:], testDataY)

print("random forest score using 3 features", rf1score)
rf2 = RandomForestRegressor()

rf2.fit(trainingDataX.drop(['left'], axis=1), trainingDataY)

rf2score = rf2.score(testDataX.drop(['left'], axis=1), testDataY)



# change testing and training data for cross validation



trainingDataY = hrData['satisfaction_level'][3000:]

trainingDataX = hrData.drop(['satisfaction_level'], axis=1)[3000:]

testDataY = hrData['satisfaction_level'][:3000]

testDataX = hrData.drop(['satisfaction_level'], axis=1)[:3000]



# Apply Second Random Forest Regression Algorithm



rf3 = RandomForestRegressor()

rf3.fit(trainingDataX.drop(['left'], axis=1), trainingDataY)

rf3score = rf2.score(testDataX.drop(['left'], axis=1), testDataY)



scoreAverage = (rf3score + rf2score) / 2



print("cross validation random forest score using all features other than left", scoreAverage)