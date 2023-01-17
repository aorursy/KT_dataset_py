import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels as stats
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
store = pd.read_csv("../input/store.csv", index_col  = 'Store')
train = pd.read_csv("../input/train.csv", index_col  = 'Store')
store.info()
train.info()
train.head(5)
%env JOBLIB_TEMP_FOLDER=/tmp
train_store_joined = train.join(store)
# replacing null values with 0
cleaned_data=train_store_joined.fillna(0)
cleaned_data.head(5)
cleaned_data.describe()
# checking linearity beetween customers visiting store vs sales
plt.scatter(cleaned_data['Customers'],cleaned_data['Sales'])
plt.grid(True)
plt.title('Customers Vs Sales', fontsize=14)
plt.xlabel('Customers', fontsize=14)
plt.ylabel('Sales', fontsize=14)
plt.show()
# The relationship between a store type and its respective assortment type
StoretypeXAssortment = sns.countplot(x="StoreType",hue="Assortment",order=["a","b","c","d"],
                                     data=cleaned_data,palette=sns.color_palette("Set2", n_colors=3)).set_title("Number of Different Assortments per Store Type")
# when are the stores open during the week?
ax = sns.countplot(x='Open', hue='DayOfWeek', data=cleaned_data, palette='Set1')
cleaned_data['StateHoliday'] = cleaned_data.StateHoliday.replace('0', 0)
cleaned_data = pd.get_dummies(cleaned_data, columns=['StateHoliday','StoreType','Assortment'], drop_first=True)
cleaned_data.info()
cleaned_data.head(5)
stats.graphics.gofplots.qqplot(cleaned_data['Promo'], line='r')
stats.graphics.gofplots.qqplot(cleaned_data['Customers'], line='r')
stats.graphics.gofplots.qqplot(cleaned_data['Promo2SinceYear'], line='r')

features = ['Customers','Open','Promo','SchoolHoliday','CompetitionDistance',
            'CompetitionOpenSinceMonth','CompetitionOpenSinceYear',
            'Promo2','Promo2SinceWeek','Promo2SinceYear','StateHoliday_a','StateHoliday_b',
            'StateHoliday_c','StoreType_b','StoreType_c','StoreType_d','Assortment_b','Assortment_c','Sales']

mask = np.zeros_like(cleaned_data[features].corr(), dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation Matrix',fontsize=25) 

sns.heatmap(cleaned_data[features].corr(),linewidths=0.25,vmax=1.0,square=True,cmap="YlGnBu", 
            linecolor='w',annot=True,mask=mask,cbar_kws={"shrink": .75})
# defining generic fuctions used for building and evaluating model
def calculate_cv_error(X,Y):
    """
    Calculates cross validation error of model
    :param X: independent variables i.e predictors contributing model
    :param Y: dependent variable i.e target in model
    :return: float value returns mean squared error
    """
    regr = linear_model.LinearRegression()
    ms_errors= cross_val_score(regr, X, Y, cv=5, scoring = make_scorer(mean_squared_error))
    rms_errors = np.sqrt(ms_errors)
    mean_rms_error = rms_errors.mean()/1000
    return mean_rms_error

def build_OLS_model(features,Y):
    """
    Build OLS linear regression model
    :param features: independent variables i.e predictors needs to be included while building model
    :param Y: dependent variable i.e target in model
    :return: Linear regression model
    """
    X = cleaned_data[features]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    return model
# building model1
features = ['Customers','Open','Promo','SchoolHoliday','CompetitionDistance',
            'CompetitionOpenSinceMonth','CompetitionOpenSinceYear',
            'Promo2','Promo2SinceWeek','Promo2SinceYear','StateHoliday_a','StateHoliday_b',
            'StateHoliday_c','StoreType_b','StoreType_c','StoreType_d','Assortment_b','Assortment_c']
X = np.array(cleaned_data[features], dtype=pd.Series)
Y = np.array(cleaned_data['Sales'], dtype=pd.Series)
print("Cross Validation Error: "+str(calculate_cv_error(X,Y)))
Y = cleaned_data['Sales']
model1 = build_OLS_model(features,Y)
results1 = model1.fit()
results1.summary()
# building model2
# As CompetitionOpenSinceYear and CompetitionOpenSinceMonth are strongly correlated removing CompetitionOpenSinceMonth
# As Promo2SinceWeek and Promo2SinceYear are strongly correlated removing Promo2SinceYear
# As Promo2 and Promo2SinceWeek are strongly correlated removing Promo2
features = ['Customers','Open','Promo','SchoolHoliday','CompetitionDistance',
            'CompetitionOpenSinceYear',
            'Promo2SinceWeek','StateHoliday_a','StateHoliday_b',
            'StateHoliday_c','StoreType_b','StoreType_c','StoreType_d','Assortment_b','Assortment_c']
X = np.array(cleaned_data[features], dtype=pd.Series)
Y = np.array(cleaned_data['Sales'], dtype=pd.Series)
print("Cross Validation Error: "+str(calculate_cv_error(X,Y)))
Y = cleaned_data['Sales']
model2 = build_OLS_model(features,Y)
results2 = model2.fit()
results2.summary()
# building model3
# As Open and StateHoliday are strongly correlated removing StateHoliday
features = ['Customers','Open','Promo','CompetitionDistance',
            'CompetitionOpenSinceYear',
            'Promo2SinceWeek','StoreType_b','StoreType_c','StoreType_d','Assortment_b','Assortment_c']
X = np.array(cleaned_data[features], dtype=pd.Series)
X = sm.add_constant(X)
Y = np.array(cleaned_data['Sales'], dtype=pd.Series)
print("Cross Validation Error: "+str(calculate_cv_error(X,Y)))
Y = cleaned_data['Sales']
model3 = build_OLS_model(features,Y)
results3 = model3.fit()
results3.summary()
# building model4
# After removing StateHoliday correlation term error increases so let go ahead with model2 predictores and add interaction term in it.
cleaned_data["Promo2SinceWeek*Promo"]=cleaned_data['Promo2SinceWeek']*cleaned_data['Promo']
cleaned_data["CompetitionDistance*CompetitionOpenSinceYear"]=cleaned_data['CompetitionDistance']*cleaned_data['CompetitionOpenSinceYear']
cleaned_data["inf"]=cleaned_data['Promo2SinceWeek']*cleaned_data['Promo']
features = ['Customers','Open','Promo','SchoolHoliday','CompetitionDistance',"CompetitionDistance*CompetitionOpenSinceYear",
            'CompetitionOpenSinceYear',"Promo2SinceWeek*Promo",
            'Promo2SinceWeek','StateHoliday_a','StateHoliday_b',
            'StateHoliday_c','StoreType_b','StoreType_c','StoreType_d','Assortment_b','Assortment_c']
X = np.array(cleaned_data[features], dtype=pd.Series)
Y = np.array(cleaned_data['Sales'], dtype=pd.Series)
print("Cross Validation Error: "+str(calculate_cv_error(X,Y)))
Y = cleaned_data['Sales']
model4 = build_OLS_model(features,Y)
results4 = model4.fit()
results4.summary()
features = ['Sales','Customers','Open','Promo','SchoolHoliday','CompetitionDistance',"CompetitionDistance*CompetitionOpenSinceYear",
            'CompetitionOpenSinceYear',"Promo2SinceWeek*Promo",
            'Promo2SinceWeek','StateHoliday_a','StateHoliday_b',
            'StateHoliday_c','StoreType_b','StoreType_c','StoreType_d','Assortment_b','Assortment_c']

dataSet=cleaned_data[features]
sales_mean= np.mean(dataSet['Sales'])
##Changing data into categorical by assigning value of sales above mean as 1 and below mean as 0
dat_classification= dataSet
dat_classification.loc[ dat_classification['Sales'] <= sales_mean, 'Sales'] = 0
dat_classification.loc[ dat_classification['Sales'] > sales_mean, 'Sales'] = 1
dat_classificationY= dat_classification['Sales']
dat_classificationX= dat_classification.drop(['Sales'],axis=1)
## Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dat_classificationX, dat_classificationY, test_size = 0.2,random_state=0)
from sklearn.ensemble import RandomForestClassifier  
randomForestCLassifier = RandomForestClassifier()  
randomForestCLassifier.fit(X_train, y_train) 
y_pred = randomForestCLassifier.predict(X_test) 
def classification_Parameters(y_test, y_pred):
    from sklearn.metrics import confusion_matrix 
    from sklearn.metrics import accuracy_score 
    from sklearn.metrics import classification_report 
    results = confusion_matrix(y_test, y_pred) 
    print ("Confusion Matrix :")
    print(results) 
    print ('Accuracy Score :',accuracy_score(y_test, y_pred))
    print ('Report : ')
    print (classification_report(y_test, y_pred)) 
classification_Parameters(y_test, y_pred)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def RocCurve(model):
    logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
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
RocCurve(randomForestCLassifier)
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
#Let us see what default parameters our model used
print('Default Parameters of :\n')
pprint(randomForestCLassifier.get_params())
from sklearn.model_selection import GridSearchCV
param_grid = {
    'bootstrap': [True],
    'max_depth': [70, 100],
    'max_features': [3, 5],
    'min_samples_leaf': [2, 4],
    'min_samples_split': [7, 10],
    'n_estimators': [200, 400]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
#Fit the grid search to the data
grid_search.fit(X_train, y_train)
grid_search.best_params_
optimizedRandom=RandomForestClassifier(n_estimators=200,max_depth=100, min_samples_split=10, min_samples_leaf=4, max_features=5,bootstrap=True)
optimizedRandom.fit(X_train,y_train)
y_pred = optimizedRandom.predict(X_test)  
classification_Parameters(y_test, y_pred)
RocCurve(optimizedRandom)
from xgboost import XGBClassifier

boostClassModel= XGBClassifier()
boostClassModel.fit(X_train, y_train)
y_pred = boostClassModel.predict(X_test) 
classification_Parameters(y_test, y_pred)
RocCurve(boostClassModel)
#Check Default parameters
boostClassModel.get_params
learning_rates = [0.5, 0.75, 1]
n_estimators=[20,30,40]
max_depths=[2,3,4]
max_features=[2,3,4]
for i in range(3):
    xgb = XGBClassifier(n_estimators=n_estimators[i], learning_rate = learning_rates[i], max_features=max_features[i], max_depth = max_depths[i], random_state = 0)
    xgb.fit(X_train, y_train)
    print("Accuracy score (training): {0:.3f}".format(xgb.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(xgb.score(X_test, y_test)))
    print()
#So best is 
#learning_rates = 1
#n_estimators=40
#max_depths=4
#max_features=4
xgb_optimized = XGBClassifier(n_estimators=40, learning_rate = 1, max_features=4, max_depth = 4, random_state = 0)
xgb_optimized.fit(X_train,y_train)
y_pred = xgb_optimized.predict(X_test)  
classification_Parameters(y_test, y_pred)
RocCurve(xgb_optimized)
from sklearn.linear_model import LogisticRegression
LRclassifier= LogisticRegression()
LRclassifier.fit(X_train,y_train)
y_pred = LRclassifier.predict(X_test)
classification_Parameters(y_test, y_pred)
RocCurve(LRclassifier)
def regessionEvaluation(y_test, y_pred):  
    from sklearn import metrics  
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
data_regression=dataSet
y= data_regression['Sales']
dataX= data_regression.drop(['Sales'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(dataX,Y, test_size=0.2, random_state=0) 
from sklearn.ensemble import RandomForestRegressor  
modelRegrRF = RandomForestRegressor()  
modelRegrRF.fit(X_train, y_train)  
y_pred = modelRegrRF.predict(X_test) 
regessionEvaluation(y_test,y_pred)
#Let us see what default parameters our model used
print('Parameters currently in use:\n')
pprint(modelRegrRF.get_params())
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
randomRegressor = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = randomRegressor, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)
#grid_search.best_params_
randomForestOptimized=RandomForestRegressor(n_estimators=200,max_depth=100, min_samples_split=10, min_samples_leaf=4, max_features=5,bootstrap=True)
randomForestOptimized.fit(X_train,y_train)
y_pred = randomForestOptimized.predict(X_test)
regessionEvaluation(y_test,y_pred)
from xgboost import XGBRegressor
xgBoostRegressor = XGBRegressor()
xgBoostRegressor.fit(X_train, y_train)
y_pred = xgBoostRegressor.predict(X_test) 
regessionEvaluation(y_test,y_pred)
learning_rates = [0.5, 0.75, 1]
n_estimators=[20,30,40]
max_depths=[2,3,4]
max_features=[2,3,4]
for i in range(3):
    xgb = XGBRegressor(n_estimators=n_estimators[i], learning_rate = learning_rates[i], max_features=max_features[i], max_depth = max_depths[i], random_state = 0)
    xgb.fit(X_train, y_train)
    print("Accuracy score (training): {0:.3f}".format(xgb.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(xgb.score(X_test, y_test)))
    print()
#So best is 
#learning_rates = 1
#n_estimators=40
#max_depths=4
#max_features=4
xgbRegressor_optimized = XGBRegressor(n_estimators=40, learning_rate = 1, max_features=4, max_depth = 4, random_state = 0)
xgbRegressor_optimized.fit(X_train,y_train)
y_pred = xgbRegressor_optimized.predict(X_test) 
regessionEvaluation(y_test,y_pred)
