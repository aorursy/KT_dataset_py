import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr

# print list of input files
print(check_output(["ls", "../input"]).decode("utf8"))

# read train.csv
dfTrainVal=pd.read_csv("../input/train.csv")
print(dfTrainVal.head(1))
print("Train data shape is",dfTrainVal.shape) # 1460, 81
print("In training data ID ranges from", dfTrainVal['Id'].describe()) # min 1 max 1460

# read test.csv
dfTest=pd.read_csv("../input/test.csv")
print(dfTest.head(1))
print("Test data shape is",dfTest.shape) # 1459, 80 (there is no 'SalesPrice')
print("In training data ID ranges from", dfTest['Id'].describe()) # min 1461, 2919

# We will need this later to split data back
numTrainValRecords = dfTrainVal.shape[0]
numTestRecords = dfTest.shape[0]
# save list of IDs in Test data
idList = dfTest['Id']
# save Y of train data
Y = dfTrainVal['SalePrice'] # type int64
# now dfTrainVal contains only X values doesn't contain SalePrice
dfTrainVal = dfTrainVal.loc[:, dfTrainVal.columns != 'SalePrice']
#### Inspection to find Categorical Data
dfTrainVal.describe()
print(dfTrainVal.describe(exclude=[np.number]))
print(dfTrainVal.dtypes)
# Above comamnds shows that the following are 43 categorical variables :
categoricalColumns = [ "MSZoning", "Street", "Alley", "LotShape", "LandContour", \
    "Utilities",  "LotConfig", "LandSlope",   "Neighborhood", "Condition1", \
    "Condition2", "BldgType", "HouseStyle",  "RoofStyle", "RoofMatl", \
    "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", \
    'Foundation', "KitchenQual",  'BsmtQual',  'BsmtCond',"BsmtExposure", \
    "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir",\
    "Electrical",  "Functional","FireplaceQu","GarageType", "GarageFinish",\
    "GarageQual","GarageCond", "PavedDrive","PoolQC", "Fence",\
    "MiscFeature","SaleType", "SaleCondition"]
# there are 2 columns of type float64
columnsOfTypefloat64 = ['LotFrontage', 'MasVnrArea'] 
# there are 29 columns of type int64
columnsOfTypeint64 = [ 'PoolArea', 'GarageArea', 'GrLivArea', 'LotArea',\
    'MSSubClass', 'OverallQual', 'OverallCond',  'GarageCars', \
    'MiscVal', '3SsnPorch', 'ScreenPorch', 'EnclosedPorch',\
    'WoodDeckSF', 'OpenPorchSF', 'Fireplaces',\
    'BsmtFullBath', 'BsmtHalfBath',  'FullBath', 'HalfBath', \
    'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', \
    'BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', \
    '1stFlrSF', '2ndFlrSF','LowQualFinSF']

# Date related columns: all other are int64 GarageYrBlt is float64
columnsRelatedToDate = [ 'MoSold', 'YrSold', 'YearBuilt', 'YearRemodAdd',  'GarageYrBlt']
# convert categorical values using one hot encoding
# refer http://pbpython.com/categorical-encoding.html
dfCombined = pd.concat([dfTrainVal, dfTest]) 
# now take actions you want to run on both test and train data
# Do not put in model : "Id" column of type int64
dfCombined.drop("Id", axis=1, inplace=True)
dfCombined = pd.get_dummies(dfCombined, columns=categoricalColumns, prefix=categoricalColumns)
# handle missing values
# refer https://www.kaggle.com/dansbecker/handling-missing-values
my_imputer = Imputer()
dfCombined = my_imputer.fit_transform(dfCombined)
targetNames = list(dfCombined)
#print(target_names)
# Scaling 
scaler = StandardScaler().fit(dfCombined)
dfCombined = pd.DataFrame(scaler.transform(dfCombined))

# Split Back test and train data
X = dfCombined.loc[0:numTrainValRecords-1,]
print("numTrainValRecords", numTrainValRecords,"X has records", X.shape[0])
XTest = dfCombined.loc[numTrainValRecords:numTrainValRecords+numTestRecords-1,]
print("numTestRecords", numTestRecords,"XTest has records", XTest.shape[0])
# http://www.blopig.com/blog/2017/07/using-random-forests-in-python-with-scikit-learn/
# simple Random forest
Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, train_size=0.75, test_size=0.25, \
                                              random_state=10)

def runRF(numEstimators, leaf_size):
    rf = RandomForestClassifier(n_estimators=numEstimators, oob_score=True, \
         n_jobs = -1, max_features = "auto", min_samples_leaf = leaf_size, random_state=10)
    rf.fit(Xtrain, Ytrain)
    predictedTrain = rf.predict(Xtrain)
    predictedVal = rf.predict(Xval)
    accuracy = accuracy_score(Yval, predictedVal)
    print("----------------------------")
    print("Random Forest with estimators", numEstimators, "leaf size", leaf_size)
    print("----------------------------")
    print(f'Mean accuracy score: {accuracy:.3}')
    test_score = r2_score(Yval, predictedVal)
    spearman = spearmanr(Yval, predictedVal)
    pearson = pearsonr(Yval, predictedVal)
    print(f'Test data R-2 score: {test_score:>5.3} Spearman correlation: {spearman[0]:.3} Pearson correlation: {pearson[0]:.3}')
    print(f'Out-of-bag score estimate: {rf.oob_score_:.3}') #>5.3
    predictedTest = rf.predict(XTest)
    return predictedTest

#sample_leaf_options = [1,5,10,50,100,200,250] # 500
#estimator_options = [50,100,200,250,500]
#for leaf_size in sample_leaf_options:
#    for n_estimator in estimator_options:
#        predictedTest = runRF(n_estimator, leaf_size)
# best oob score was with 50 and 50
predictedTestBest = runRF(50, 50)

#pca_scaled = PCA() #pca = PCA(n_components=20)
#train_features = pca_scaled.fit_transform(Xtrain, YTrain)
#test_features = pca.transform(Xval, Yval)
#x_axis = np.arange(1, pca.n_components_+1)
#predictedTestPCA = runRF(50, 50)

# Write output to file
import csv
with open('output.csv', 'w', newline='') as csvfile:
    fieldnames = ['Id', 'SalePrice']    
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for index in range(len(idList)):
        writer.writerow({'Id': idList[index], 'SalePrice': predictedTestBest[index]})
print("End")