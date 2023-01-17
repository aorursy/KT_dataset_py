#pandas is an open-source library that provides high-performance, easy-to-use data structures and data analysis tools
#pandas adds data analysis and modeling tools so that users can perform entire data science workflows.
import pandas as pd
#There are several Python libraries which provide solid implementations of a range of machine learning algorithms. 
#One of the best known is Scikit-Learn(Python module), a package that provides efficient versions of a large number of common algorithms.


#The goal of ensemble methods is to combine the predictions of several base estimators built with a given learning algorithm 
#in order to improve generalizability / robustness over a single estimator.
#Two families of ensemble methods:
# 1-averaging methods, the driving principle is to build several estimators independently and then to average their predictions
#On average, the combined estimator is usually better than any of the single base estimator because its variance is reduced.
#Bagging methods, Forests of randomized trees, …
#2-by contrast, in boosting methods, base estimators are built sequentially and one tries to reduce the bias of the combined estimator.
#The motivation is to combine several weak models to produce a powerful ensemble.
from sklearn.ensemble import RandomForestRegressor
TrainData = pd.read_csv('../input/train.csv')
# here i want some parameters that are in PredictorColumns to determine/predict sale price which here is train_y
Train_y = TrainData.SalePrice
PredictorColumns = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
Train_x = TrainData[PredictorColumns]
MyModel  = RandomForestRegressor()
#Fit: Capture patterns from provided data. This is the heart of modeling.
MyModel.fit(Train_x, Train_y)

# to test our train data we need test data or we can split train data (but not that good option)
TestData = pd.read_csv('../input/test.csv')
Test_x = TestData[PredictorColumns]
PredictedPrices = MyModel.predict(Test_x)
#here is the way i want my output excel file to look like 
MySubmission = pd.DataFrame ({'Id': TestData.Id, 'SalePrice': PredictedPrices })
# index false so that csv file dont have extra coloumns 
MySubmission.to_csv('SalmaFirstCompetitionHousePrices.csv', index=False)
