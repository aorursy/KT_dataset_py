import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
CancerData = pd.read_csv('/kaggle/input/cancer_reg.csv', encoding='latin')
print('Shape before removing duplicates', CancerData.shape)
CancerData = CancerData.drop_duplicates()
print('Shape before removing duplicates', CancerData.shape)
CancerData.head()
CancerData.columns
CancerData = CancerData[['avgAnnCount', 'avgDeathsPerYear', 'incidenceRate',
       'medIncome', 'popEst2015', 'povertyPercent', 'studyPerCap', 'binnedInc',
       'MedianAge', 'MedianAgeMale', 'MedianAgeFemale', 'Geography',
       'AvgHouseholdSize', 'PercentMarried', 'PctNoHS18_24', 'PctHS18_24',
       'PctSomeCol18_24', 'PctBachDeg18_24', 'PctHS25_Over',
       'PctBachDeg25_Over', 'PctEmployed16_Over', 'PctUnemployed16_Over',
       'PctPrivateCoverage', 'PctPrivateCoverageAlone', 'PctEmpPrivCoverage',
       'PctPublicCoverage', 'PctPublicCoverageAlone', 'PctWhite', 'PctBlack',
       'PctAsian', 'PctOtherRace', 'PctMarriedHouseholds', 'BirthRate', 'TARGET_deathRate']]
CancerData.head()
CancerData.describe()
CancerData.info()
CancerData.isna().sum()
CancerData.drop(['PctPrivateCoverageAlone', 'PctSomeCol18_24'], axis=1, inplace=True)
CancerData.isna().sum()
CancerData.iloc[1:10,7:]
CancerData.nunique()
CancerData.drop('Geography', axis=1, inplace=True)
CancerData.shape
CancerData.head()
%matplotlib inline
histo=CancerData.hist(['incidenceRate', 'medIncome',
       'povertyPercent', 'MedianAgeMale', 'MedianAgeFemale', 'AvgHouseholdSize',
       'PercentMarried', 'PctNoHS18_24', 'PctHS18_24', 'PctBachDeg18_24'], figsize=(30,30))
histo = CancerData.hist(['PctHS25_Over', 'PctBachDeg25_Over', 'PctEmployed16_Over',
       'PctUnemployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage',
       'PctPublicCoverage', 'PctPublicCoverageAlone', 'PctWhite', 'PctBlack',
       'PctOtherRace', 'PctMarriedHouseholds', 'BirthRate',
       'TARGET_deathRate'], figsize=(30,30))
['incidenceRate', 'medIncome', 'povertyPercent', 'MedianAgeMale', 'MedianAgeFemale', 'AvgHouseholdSize',
'PercentMarried', 'PctNoHS18_24', 'PctHS18_24', 'PctBachDeg18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 'PctEmployed16_Over',
'PctUnemployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage',
'PctPublicCoverage', 'PctPublicCoverageAlone', 'PctWhite', 'PctBlack',
'PctOtherRace', 'PctMarriedHouseholds', 'BirthRate',
'TARGET_deathRate']
import matplotlib.pyplot as plt
CancerData.groupby('binnedInc').size().plot.bar()
scatter = pd.plotting.scatter_matrix(CancerData, figsize=(20,20))
def ConVSCon(inpData, Cols, Target):
    fig,subplot = plt.subplots(nrows = len(Cols), ncols = 1, figsize = (5,80))
    for ColName, PlotNumber in zip(Cols, range(len(Cols))):
        inpData.plot.scatter(x = ColName, y = Target, ax = subplot[PlotNumber])
ConVSCon(inpData=CancerData, Cols=['incidenceRate', 'medIncome', 'povertyPercent', 'MedianAgeMale', 'MedianAgeFemale', 'AvgHouseholdSize',
'PercentMarried', 'PctNoHS18_24', 'PctHS18_24', 'PctBachDeg18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 'PctEmployed16_Over',
'PctUnemployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage',
'PctPublicCoverage', 'PctPublicCoverageAlone', 'PctWhite', 'PctBlack',
'PctOtherRace', 'PctMarriedHouseholds', 'BirthRate'], Target='TARGET_deathRate')
plt.scatter(x=CancerData['incidenceRate'], y=CancerData['TARGET_deathRate'])
CancerData.plot.scatter(x = 'incidenceRate', y='TARGET_deathRate',marker='o')
CoorData = CancerData.corr()
np.abs(CoorData['TARGET_deathRate']).sort_values(ascending=False)
['TARGET_deathRate', 'PctBachDeg25_Over', 'incidenceRate',
       'PctPublicCoverageAlone', 'povertyPercent', 'medIncome',
       'PctEmployed16_Over', 'PctHS25_Over', 'PctPublicCoverage',
       'PctPrivateCoverage', 'PctUnemployed16_Over', 'PctMarriedHouseholds',
       'PctBachDeg18_24', 'PctEmpPrivCoverage', 'PercentMarried', 'PctHS18_24',
       'PctBlack']
bar = CancerData.groupby(['binnedInc']).mean()['TARGET_deathRate'].plot.bar(figsize = (5,5))
CancerData.boxplot(by='binnedInc', column='TARGET_deathRate', figsize=(10,10))
from scipy.stats import f_oneway
categrpList = CancerData.groupby(['binnedInc'])['TARGET_deathRate'].apply(list)
ANOVA = f_oneway(*categrpList)
print(np.round(ANOVA[1],decimals = 10))
CancerData.info()
MapResult = {'(61494.5, 125635]':10, '(48021.6, 51046.4]':7, '(42724.4, 45201]':5,
       '(51046.4, 54545.6]':8, '(37413.8, 40362.7]':3, '(40362.7, 42724.4]':4,
       '(54545.6, 61494.5]':9, '(34218.1, 37413.8]':2, '[22640, 34218.1]':1,
       '(45201, 48021.6]':6}
CancerData['binnedInc'] = CancerData['binnedInc'].map(MapResult)
CancerData.head()
CancerData['binnedInc'].unique()
CancerData.isna().sum()
CancerData['PctEmployed16_Over'] = CancerData['PctEmployed16_Over'].fillna(CancerData['PctEmployed16_Over'].median())
CancerData['PctEmployed16_Over'].isna().sum()
Predictors = ['PctBachDeg25_Over', 'incidenceRate',
       'PctPublicCoverageAlone', 'povertyPercent', 'medIncome',
       'PctEmployed16_Over', 'PctHS25_Over', 'PctPublicCoverage',
       'PctPrivateCoverage', 'PctUnemployed16_Over', 'PctMarriedHouseholds',
       'PctBachDeg18_24', 'PctEmpPrivCoverage', 'PercentMarried', 'PctHS18_24',
       'PctBlack', 'binnedInc']
Target = ['TARGET_deathRate']
X = CancerData[Predictors].values
y = CancerData[Target].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lrRegModel = lr.fit(X_train, y_train)
lrPrediction = lrRegModel.predict(X_test)
from sklearn.metrics import r2_score
print('R2 value', r2_score(y_train, lrRegModel.predict(X_train)))
print('Accuracy', 100-(np.mean((np.abs(y_test-lrPrediction)/y_test))*100))

TestingData = pd.DataFrame(X_test, columns=Predictors)
TestingData['Target'] = y_test
TestingData['PredictedValue'] = lrPrediction
TestingData['APE'] = (np.abs(y_test-lrPrediction)/y_test)*100
TestingData.head()
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=6, criterion='mse')
dtRegModel = dt.fit(X_train, y_train)
dtPrediction = dtRegModel.predict(X_test)
print('R2 value', r2_score(y_train, dtRegModel.predict(X_train)))
print('Accuracy', 100-(np.mean((np.abs(y_test-dtPrediction)/y_test))*100))

feature_importances = pd.Series(dtRegModel.feature_importances_, index=Predictors)
feature_importances.nlargest(17).plot.barh()

TestingData = pd.DataFrame(X_test, columns=Predictors)
TestingData['Target'] = y_test
TestingData['PredictedValue'] = dtPrediction
#TestingData['APE'] = (np.abs(y_test-dtPrediction)/y_test)*100
TestingData.head()
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=1, criterion='mse', n_estimators=100)
rfRegModel = rf.fit(X_train, y_train)
rfPrediction = rfRegModel.predict(X_test)
print('R2 value', r2_score(y_train, rfRegModel.predict(X_train)))
print('Accuracy', 100-(np.mean((np.abs(y_test-rfPrediction)/y_test))*100))

feature_importances = pd.Series(rfRegModel.feature_importances_, index=Predictors)
feature_importances.nlargest(17).plot.barh()

TestingData = pd.DataFrame(X_test, columns=Predictors)
TestingData['Target'] = y_test
TestingData['PredictedValue'] = rfPrediction
#TestingData['APE'] = (np.abs(y_test-rfPrediction)/y_test)*100
TestingData.head()
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=1)
adab = AdaBoostRegressor(n_estimators=100, base_estimator=dt, learning_rate=0.1)
adabRegModel = adab.fit(X_train, y_train)
adabPrediction = adabRegModel.predict(X_test)
print('R2 value', r2_score(y_train, adabRegModel.predict(X_train)))
print('Accuracy', 100-(np.mean((np.abs(y_test-adabPrediction)/y_test))*100))

feature_importances = pd.Series(adabRegModel.feature_importances_, index=Predictors)
feature_importances.nlargest(17).plot.barh()

TestingData = pd.DataFrame(X_test, columns=Predictors)
TestingData['Target'] = y_test
TestingData['PredictedValue'] = adabPrediction
#TestingData['APE'] = (np.abs(y_test-rfPrediction)/y_test)*100
TestingData.head()
from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.2, booster='gbtree')
xgbRegModel = xgb.fit(X_train, y_train)
xgbPrediction = xgbRegModel.predict(X_test)
print('R2 value', r2_score(y_train, xgbRegModel.predict(X_train)))
print('Accuracy', 100-(np.mean((np.abs(y_test-xgbPrediction)/y_test))*100))

feature_importances = pd.Series(xgbRegModel.feature_importances_, index=Predictors)
feature_importances.nlargest(17).plot.barh()

TestingData = pd.DataFrame(X_test, columns=Predictors)
TestingData['Target'] = y_test
TestingData['PredictedValue'] = xgbPrediction
#TestingData['APE'] = (np.abs(y_test-rfPrediction)/y_test)*100
TestingData.head()
# Linear Regression
def LinearRegressionParams(X_train, y_train, X_test, y_test):
    test_size_list = [0.2,0.25,0.3]
    random_state_list = [42,775,687]
    TrialNo = 0
    for Test_size in test_size_list:
        for Random_state in random_state_list:
            TrialNo+=1
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = Test_size, random_state = Random_state)
            lr = LinearRegression()
            lrRegModel = lr.fit(X_train, y_train)
            lrPrediction = lrRegModel.predict(X_test)
            print(TrialNo, 'Random_state', Random_state, '--> Test_size', Test_size, '--> Accuracy',100-(np.mean((np.abs(y_test-lrPrediction)/y_test))*100))
LinearRegressionParams(X_train, y_train, X_test, y_test)
# AdaBoost
def AdaboostParams(X_train, y_train, X_test, y_test):
    test_size_list = [0.2,0.25,0.3]
    random_state_list = [42,775,687]
    N_Estimators_list = [500, 550, 600]
    TrialNo = 0
    for Test_size in test_size_list:
        for Random_state in random_state_list:
            for N_Estimators in N_Estimators_list:
                TrialNo+=1
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = Test_size, random_state = Random_state)
                adab = AdaBoostRegressor(n_estimators=N_Estimators, base_estimator=dt, learning_rate= 0.2)
                adabRegModel = adab.fit(X_train, y_train)
                adabPrediction = adabRegModel.predict(X_test)
                Accuracy = 100-(np.mean((np.abs(y_test-adabPrediction)/y_test))*100)
                print(TrialNo, 'Random_state', Random_state,
                      '--> Test_size', Test_size, 'n_estimators',N_Estimators, '--> Accuracy', Accuracy)
AdaboostParams(X_train, y_train, X_test, y_test)
# Random Forest
def RandomForestParams(X_train, y_train, X_test, y_test):
    test_size_list = [0.2,0.25,0.3]
    random_state_list = [42,775,687]
    N_Estimators_list = [500, 550, 600]
    TrialNo = 0
    for Test_size in test_size_list:
        for Random_state in random_state_list:
            for N_Estimators in N_Estimators_list:
                TrialNo+=1
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = Test_size, random_state = Random_state)
                rf = RandomForestRegressor(max_depth=1, criterion='mse', n_estimators=N_Estimators)
                rfRegModel = rf.fit(X_train, y_train)
                rfPrediction = rfRegModel.predict(X_test)
                Accuracy = 100-(np.mean((np.abs(y_test-rfPrediction)/y_test))*100)
                print(TrialNo, 'Random_state', Random_state,
                      '--> Test_size', Test_size, 'n_estimators',N_Estimators, '--> Accuracy', Accuracy)
RandomForestParams(X_train, y_train, X_test, y_test)
# Importing Layers and Models
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units = 5, input_dim = 17, kernel_initializer = 'normal', activation = 'relu'))
model.add(Dense(units = 20, kernel_initializer = 'normal', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'normal'))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(X_train, y_train, verbose=1, batch_size=10, epochs=20)
def ANNBestParams(X_train, y_train, X_test, y_test):
    test_size_list = [0.2,0.25,0.3]
    Batch_size_list = [5,10,15]
    Epochs_list = [10,50,100]
    TrialNo = 0
    for test_size in test_size_list:
        for Batch_size in Batch_size_list:
            for Epochs in Epochs_list:
                TrialNo+=1
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
                model = Sequential()
                model.add(Dense(units = 5, input_dim = 17, kernel_initializer = 'normal', activation = 'relu'))
                model.add(Dense(units = 20, kernel_initializer = 'normal', activation = 'relu'))
                model.add(Dense(units = 1, kernel_initializer = 'normal'))
                model.compile(loss = 'mean_squared_error', optimizer = 'adam')
                model.fit(X_train, y_train, verbose=0, batch_size=Batch_size, epochs=Epochs)
                Prediction = model.predict(X_test)
                print(TrialNo, '--> Test_size', test_size, 'batch_size', Batch_size, 'epochs', Epochs,
                      '--> Accuracy',100-(np.mean((np.abs(y_test-Prediction)/y_test))*100))
ANNBestParams(X_train, y_train, X_test, y_test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
model.fit(X_train, y_train, epochs=50, batch_size=15, verbose=0)
ANNPrediction = model.predict(X_test)
TestingData = pd.DataFrame(X_test, columns=Predictors)
TestingData['TARGET_deathRate'] = y_test
TestingData['Predicted_deathRate'] = ANNPrediction
TestingData.head()
100-(np.mean((np.abs(TestingData['TARGET_deathRate']-TestingData['Predicted_deathRate'])/TestingData['TARGET_deathRate']))*100)

