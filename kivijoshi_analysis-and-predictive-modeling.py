import pandas as pd



Gym_data = pd.read_csv('../input/data.csv')

Gym_data.head()
Gym_data.describe()
%matplotlib inline



import matplotlib.pyplot as plt

import seaborn as sb

sb.pairplot(Gym_data)
corrmat = Gym_data.corr()

f, ax = plt.subplots(figsize=(7, 7))

# Draw the heatmap using seaborn

sb.heatmap(corrmat, square=False)

plt.show()
Bins = []

for i in range(0,24):

    NumberofPeople = 0

    for index, row in Gym_data.iterrows():

        if(row['timestamp']/3600 > i and row['timestamp']/3600 < i+1):

            NumberofPeople = NumberofPeople + row['number_people']

    Bins.append((NumberofPeople))

                  

sb.barplot(list(range(24)),Bins)
Bins = []

for i in range(0,7):

    NumberofPeople = 0

    for index, row in Gym_data.iterrows():

        if(row['day_of_week'] >= i and row['day_of_week'] < i+1):

            NumberofPeople = NumberofPeople + row['number_people']

    Bins.append((NumberofPeople))

print(Bins)

            

                       

sb.barplot(list(range(7)),Bins)
Bins = []

for i in range(30,100,10):

    NumberofPeople = 0

    for index, row in Gym_data.iterrows():

        if(row['temperature'] >= i and row['temperature'] < i+10):

            NumberofPeople = NumberofPeople + row['number_people']

    Bins.append((NumberofPeople))

                  

sb.barplot(list(range(30,100,10)),Bins)
sb.distplot(Gym_data['temperature'], kde=False, rug=True)
NumberofPeopleDuringSem = 0

NumberofPeoplestartSem = 0

for index, row in Gym_data.iterrows():

    if(row['is_start_of_semester'] == 0):

        NumberofPeopleDuringSem = NumberofPeopleDuringSem + row['number_people']

    else:

        NumberofPeoplestartSem = NumberofPeoplestartSem + row['number_people']

print('Number of people at start of sem = ' + str(NumberofPeoplestartSem))

print('Number of people during sem = ' + str(NumberofPeopleDuringSem))
from sklearn.linear_model import SGDClassifier,SGDRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

from sklearn.calibration import CalibratedClassifierCV

#import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn import svm
Gym_data_copy = Gym_data

y = Gym_data_copy['number_people'].values

Gym_data_copy = Gym_data_copy.drop(['number_people','apparent_temperature','is_weekend'],axis=1)

X = Gym_data_copy.values

Gym_data_copy.head()
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.50)
# Scale the data to be between -1 and 1

scaler = StandardScaler()

scaler.fit(Xtrain)

Xtrain = scaler.transform(Xtrain)

Xtest = scaler.transform(Xtest)
sdg = SGDRegressor()

sdg.fit(Xtrain, ytrain)

y_val_l = sdg.predict(Xtest)

print(sdg.score(Xtest, ytest))
radm = RandomForestRegressor()

radm.fit(Xtrain, ytrain)

y_val_l = radm.predict(Xtest)

print(radm.score(Xtest, ytest))
import numpy as np

indices = np.argsort(radm.feature_importances_)[::-1]



# Print the feature ranking

print('Feature ranking:')



for f in range(Gym_data_copy.shape[1]):

    print('%d. feature %d %s (%f)' % (f+1 , indices[f], Gym_data_copy.columns[indices[f]],

                                      radm.feature_importances_[indices[f]]))