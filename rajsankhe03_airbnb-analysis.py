from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://fortunedotcom.files.wordpress.com/2014/07/new-logos-airbnb.jpg")
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import seaborn as sns
import statsmodels.api as sm
import statsmodels as statm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from plotnine import *

filedata= pd.read_csv('../input/train.csv')
filedata.head(5)
len(filedata.columns)
filedata.shape
filedata.info()
#check for missing data, and output columns that have missing data
for col in filedata:
    if (filedata[col].isnull().any()):
        print(col)
#fills missing data with 0s
#GO BACK TO THIS, 0 may not be best fill for all missing data
filedata=filedata.fillna(0)
#summary stats on each of the numeric columns
filedata.describe()
#check all the statistics
filedata.describe(include='all')
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','uint8']
numericdataX = filedata.select_dtypes(include=numerics)
x= numericdataX['accommodates']
sns.distplot(x);
x= numericdataX.iloc[:,1]
sns.distplot(x);
ggplot(filedata, aes(x='room_type')) + geom_bar(fill = "red")
ggplot(filedata, aes(x='city')) + geom_bar(fill = "green")
#check categorical data
filedata.describe(include=['O'])
#check numeric data
filedata.describe()
filedata.columns
regressor = linear_model.LinearRegression()
for i in range(1,10): 
    x= np.array(numericdataX.iloc[:,i]).reshape(-1,1)
    y= np.array(filedata['log_price']).reshape(-1,1)
    regressor.fit(x,y)
    plt.figure(figsize=(8,5))
    plt.subplot(10,1,i)
    plt.scatter(x,y,color='blue', alpha=0.1)
    plt.plot(x,regressor.predict(x),color="red")
    plt.legend()
statm.graphics.gofplots.qqplot(numericdataX.iloc[:,6], line='r')
statm.graphics.gofplots.qqplot(numericdataX.iloc[:,1], line='r')
statm.graphics.gofplots.qqplot(numericdataX.iloc[:,9], line='r')
def checkCorrelation(data):
    """
    Plot correlation Matrix for given data
   :param data: dataset having features
   :return: return plot representing pearson correlation
   """
    plt.figure(figsize=(20, 20))
    sns.heatmap(data.corr(),linewidths=0.25,vmax=1.0,square=True,cmap="BuGn_r", 
    linecolor='w',annot=True)
#return Model 
def data_model(xdata):
    """
     fits linear regression model on given data
    :param xdata: independent variable dataset
    :return: linear regression model with fit of xdata 
   """
    #add constant to data
    X = sm.add_constant(xdata)
    targetY=filedata[['log_price']]
    y = targetY

    # Fit the linear model
    model = linear_model.LinearRegression()
    results = model.fit(X, y)
    model = sm.OLS(y, X)
    results = model.fit()
    return results
def data_summary(xdata):
    """
    Returns chart having summary of data
   :param xdata: independent variable dataset
   :return: summary of data 
   """
    results = data_model(xdata)
    return results.summary()
def crossValidationError(data):
    """
   Finds cross validation error of model
   :param X: independent variable dataset
   :return: float value returns mean squared error
   """
    numericdataX=data
    X = np.array(numericdataX.drop(['log_price'],axis=1), dtype=pd.Series)
    Y = np.array(numericdataX['log_price'], dtype=pd.Series)
    regr1 = linear_model.LinearRegression()
    ms_errors= cross_val_score(regr1, X, Y, cv=5, scoring = make_scorer(mean_squared_error))
    rms_errors = np.sqrt(ms_errors)
    mean_rms_error = rms_errors.mean()
    return mean_rms_error
#Checking correlation in data
checkCorrelation(numericdataX)
#So as per correlation matrix colums such as latitude, longitude, number_of_reviews and review_scores_rating are not making much impact on log_price
#as valueof cirrelation is poor
#lets drop them from our dataset
numericdataX=numericdataX.drop(['id','number_of_reviews',
       'review_scores_rating','latitude',
       'longitude' ], axis=1)
# buid model and check summary
data_summary(numericdataX)
# there is also correlation between bathroom and accomodates and bedroom lets only keep acomodates
numericdataX = numericdataX.drop(['bathrooms','bedrooms','beds'], axis=1)
# buid model and check summary
data_summary(numericdataX)
crossValidationError(numericdataX)
filedata.room_type.value_counts()
#creating dummy variable for column room_type
numericdataX=pd.concat([numericdataX,filedata['room_type']], axis=1)
numericdataX=pd.get_dummies(numericdataX,columns= ['room_type'],drop_first=True)
numericdataX
filedata.bed_type.value_counts()
numericdataX=pd.concat([numericdataX,filedata['bed_type']], axis=1)
numericdataX=pd.get_dummies(numericdataX,columns=['bed_type'],drop_first=True)
filedata.cancellation_policy.value_counts()
numericdataX=pd.concat([numericdataX,filedata['cancellation_policy']], axis=1)
numericdataX=pd.get_dummies(numericdataX,columns=['cancellation_policy'],drop_first=True)
filedata.city.value_counts()
numericdataX=pd.concat([numericdataX,filedata['city']], axis=1)
numericdataX=pd.get_dummies(numericdataX,columns=['city'],drop_first=True)
filedata.instant_bookable.value_counts()
numericdataX=pd.concat([numericdataX,filedata['instant_bookable']], axis=1)
numericdataX=pd.get_dummies(numericdataX,columns=['instant_bookable'],drop_first=True)
checkCorrelation(numericdataX)
data_summary(numericdataX.drop(['log_price'],axis=1))
filedata.property_type.value_counts()
numericdataX=pd.concat([numericdataX,filedata['property_type']], axis=1)
numericdataX=pd.get_dummies(numericdataX,columns=['property_type'],drop_first=True)
data_summary(numericdataX)
crossValidationError(numericdataX)
# P value of bed type has poor P value
numericdataX = numericdataX.loc[:, ~numericdataX.columns.str.startswith('bed_type_')]
data_summary(numericdataX.drop(['log_price'],axis=1))
crossValidationError(numericdataX)
filedata.columns
interactionDF= pd.DataFrame()
interactionDF['bedrooms']=filedata['bedrooms']
interactionDF['beds']=filedata['beds']
interactionDF['bathrooms']=filedata['bathrooms']
interactionDF['bed*bathroom*bedrooms']=filedata['bedrooms']*filedata['beds']*filedata['bathrooms']
data_summary(interactionDF)
numericdataX= pd.concat([numericdataX,interactionDF],axis=1)
data_summary(numericdataX)
interactionDF1= pd.DataFrame()
interactionDF1['review_scores_rating']=filedata['review_scores_rating']
interactionDF1['number_of_reviews']=filedata['number_of_reviews']
interactionDF1['reiew_score*Number']=filedata['review_scores_rating']*filedata['number_of_reviews']
data_summary(interactionDF1)
numericdataX= pd.concat([numericdataX,interactionDF1],axis=1)
data_summary(numericdataX)
crossValidationError(numericdataX)

##import h2o
##from h2o.automl import H2OAutoML
##h2o.init()
numericdataX
filedata
mean_log= np.mean(numericdataX['log_price'])
classificationData= numericdataX
classificationData.loc[ classificationData['log_price'] <= mean_log, 'log_price'] = 0
classificationData.loc[ classificationData['log_price'] > mean_log, 'log_price'] = 1
classificationDataY= classificationData['log_price']
classificationDataX=classificationData.drop(['log_price'],axis=1)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(classificationDataX, classificationDataY, test_size = 0.2,random_state=0)
classifier= LogisticRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def rocAucCurve(classifier):
    logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
rocAucCurve(classifier)
from sklearn.ensemble import RandomForestClassifier  
classifierDT = RandomForestClassifier()  
classifierDT.fit(X_train, y_train) 
y_pred = classifierDT.predict(X_test)  
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  
rocAucCurve(classifierDT)
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
#Let us see what default parameters our model used
print('Parameters currently in use:\n')
pprint(classifierDT.get_params())
from sklearn.model_selection import GridSearchCV
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4],
    'min_samples_split': [8, 10],
    'n_estimators': [100, 200]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_params_
random1=RandomForestClassifier(n_estimators=200,max_depth=90, min_samples_split=8, min_samples_leaf=3, max_features=3,bootstrap=True)
random1.fit(X_train,y_train)
y_pred = random1.predict(X_test)  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
rocAucCurve(random1)
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)  
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred)) 
rocAucCurve(gb)
learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)
    gb.fit(X_train, y_train)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_test, y_test)))
    print()
#Learning rate 1  is good 
gb_op = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.75, max_features=2, max_depth = 2, random_state = 0)
gb_op.fit(X_train,y_train)
y_pred = gb_op.predict(X_test)  
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred)) 
rocAucCurve(gb_op)
Y= filedata['log_price']
numericdataX= numericdataX.drop(['log_price'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(numericdataX,Y, test_size=0.2, random_state=0)  
from sklearn.ensemble import RandomForestRegressor  
regressor = RandomForestRegressor()  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test) 
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})  
df
def regression_Metrics(y_test, y_pred):  
    from sklearn import metrics  
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
regression_Metrics(y_test,y_pred)
#Let us see what default parameters our model used
print('Parameters currently in use:\n')
pprint(regressor.get_params())
from sklearn.model_selection import GridSearchCV
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4],
    'min_samples_split': [8, 10],
    'n_estimators': [100, 200]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(X_train, y_train)
grid_search.best_params_
regressor1=RandomForestRegressor(n_estimators=200,max_depth=90, min_samples_split=10, min_samples_leaf=10, max_features=3,bootstrap=True)
regressor1.fit(X_train,y_train)
y_pred = regressor1.predict(X_test) 
regression_Metrics(y_test,y_pred)
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)  
regression_Metrics(y_test,y_pred)
learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    gb = GradientBoostingRegressor(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)
    gb.fit(X_train, y_train)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_test, y_test)))
    print()
gb_op = GradientBoostingRegressor(n_estimators=20, learning_rate = 1, max_features=2, max_depth = 2, random_state = 0)
gb_op.fit(X_train,y_train)
y_pred = gb_op.predict(X_test)
regression_Metrics(y_test,y_pred)
