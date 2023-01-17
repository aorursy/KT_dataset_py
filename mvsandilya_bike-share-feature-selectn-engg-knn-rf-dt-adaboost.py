# import relevant libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
bike = pd.read_csv('../input/train.csv')
# Check for missing values - AWESOME!! no missing values in the "Training" dataset
# The heatmap tool of seaborn can also be used for a visual representation for checking missing values
bike.info()
# check out the head to see categorical & quant variables and initial observations
bike.head()
bike['datetime'].head()
import datetime
bike['datetime'].iloc[0]
bike['datetime'] = pd.to_datetime(bike['datetime'])
bike['Hour'] = bike['datetime'].apply(lambda time: time.hour)
bike['Month'] = bike['datetime'].apply(lambda time: time.month)
bike['year'] = bike['datetime'].apply(lambda time:time.year)
bike['Day of Week'] = bike['datetime'].apply(lambda time: time.dayofweek)
bike.head()
sns.countplot(x='Day of Week',data=bike,palette='viridis')
plt.scatter('Hour', 'count', data = bike)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# How do we select the 4 best predictive features out of the feature set available?
df = bike.drop('datetime', axis = 1)
X = df.drop('count', axis = 1)
#Example = yourdf.drop(['columnheading1', 'columnheading2'], axis=1, inplace=True)
y = df['count']
test = SelectKBest(score_func = chi2, k = 4) # Instantiating selectkbest 
fit = test.fit(X,y)  # Now we fit selectkbest to the data
print(fit.scores_)
features = fit.transform(X)
print(features[0:5,:]) # higher the score, better the rating
plt.figure(figsize = (18,18))
sns.heatmap(bike.corr(), cmap='coolwarm', annot = True)
select_features = bike[['temp', 'casual', 'registered', 'Hour', 'humidity']]
select_features.head()
sns.distplot(bike['temp'], bins = 10) # we'll assume a normal distribution and move ahead  
sns.countplot(x='Month',data=bike,palette='viridis')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(select_features,bike['count'],
                                                    test_size=0.30, random_state=101)
from sklearn.ensemble import RandomForestRegressor
rfc = RandomForestRegressor(n_estimators=100)
rfc.fit(X_train, y_train) # This has been run on scaled features
rfc_pred = rfc.predict(X_test)
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, rfc_pred))
rms
from sklearn.preprocessing import StandardScaler # need to scale feature set to fit KNN
scaler = StandardScaler() # initialise a scaler object to run on a dataframe
select_features = bike[['temp', 'casual', 'registered', 'Hour', 'humidity']]
scaler.fit(select_features) # run the above scaler method on the selected dataframe
scaled_features = scaler.transform(select_features)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(select_features,bike['count'],
                                                    test_size=0.30, random_state=101)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20) # initialise the KNN classifier with neighbours=20 in this case
knn.fit(X_train,y_train)
pred = knn.predict(X_test) #run the KNN model on the test data
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, pred))
rms
select_features = bike[['temp', 'casual', 'registered', 'Hour', 'humidity']] # Adding year though it has a low correlation with the target variable - 'count'
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(select_features,bike['count'],
                                                    test_size=0.30, random_state=101)
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor() 
clf = AdaBoostRegressor(n_estimators=100, base_estimator=dt,learning_rate=1)
#Above I have used decision tree as a base estimator, you can use any ML learner as base estimator if it accepts sample weight 
clf.fit(X_train,y_train)
clf_pred = clf.predict(X_test)
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, clf_pred))
rms
from sklearn import metrics
print (metrics.accuracy_score(y_test, clf_pred))
bike['count'].describe()
sns.distplot(bike['count'], bins = 10)