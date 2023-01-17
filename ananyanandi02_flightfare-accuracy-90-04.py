import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
import seaborn as sns
train = pd.read_excel('../input/flight-fare-prediction-mh/Data_Train.xlsx')
test= pd.read_excel('../input/flight-fare-prediction-mh/Test_set.xlsx')
train.shape
test.shape
train.head(5)
print(train.dtypes)
print(test.dtypes)
print(train.isnull().sum()) #checking for null values
print(test.isnull().sum())
train=train.dropna()
train.drop_duplicates()
train.shape
train['Journey_Day'] = pd.to_datetime(train.Date_of_Journey, format='%d/%m/%Y').dt.day
train['Journey_Month'] = pd.to_datetime(train.Date_of_Journey, format='%d/%m/%Y').dt.month
train['weekday']= pd.to_datetime(train.Date_of_Journey, format='%d/%m/%Y').dt.weekday

test['Journey_Day'] = pd.to_datetime(test.Date_of_Journey, format='%d/%m/%Y').dt.day
test['Journey_Month'] = pd.to_datetime(test.Date_of_Journey, format='%d/%m/%Y').dt.month
test['weekday']= pd.to_datetime(test.Date_of_Journey, format='%d/%m/%Y').dt.weekday
train.drop(labels = 'Date_of_Journey', axis = 1, inplace = True)
test.drop(labels = 'Date_of_Journey', axis = 1, inplace = True)
train.columns
def duration(test):
    test = test.strip()
    total=test.split(' ')
    to=total[0]
    hrs=(int)(to[:-1])*60
    if((len(total))==2):
        mint=(int)(total[1][:-1])
        hrs=hrs+mint
    test=str(hrs)
    return test
train['Duration']=train['Duration'].apply(duration)
test['Duration']=test['Duration'].apply(duration)
def deparrtime(x):
    x=x.strip()
    tt=(int)(x.split(':')[0])
    if(tt>=16 and tt<21):
        x='Evening'
    elif(tt>=21 or tt<5):
        x='Night'
    elif(tt>=5 and tt<11):
        x='Morning'
    elif(tt>=11 and tt<16):
        x='Afternoon'
    return x
train['Dep_Time']=train['Dep_Time'].apply(deparrtime)
test['Dep_Time']=test['Dep_Time'].apply(deparrtime)
train['Arrival_Time']=train['Arrival_Time'].apply(deparrtime)
test['Arrival_Time']=test['Arrival_Time'].apply(deparrtime)
def stops(x):
    if(x=='non-stop'):
        x=str(0)
    else:
        x.strip()
        stps=x.split(' ')[0]
        x=stps
    return x
train['Total_Stops']=train['Total_Stops'].apply(stops)
test['Total_Stops']=test['Total_Stops'].apply(stops)
pd.options.mode.chained_assignment = None 
for i in range(train.shape[0]):
    if(train.iloc[i]['Additional_Info']=='No info'):
        train.iloc[i]['Additional_Info']='No Info' 
pd.options.mode.chained_assignment = None 
for i in range(test.shape[0]):
    if(test.iloc[i]['Additional_Info']=='No info'):
        test.iloc[i]['Additional_Info']='No Info' 
train=train.drop(['Route'], axis=1) #we don't need it as we already have total_stops
test=test.drop(['Route'], axis=1)
train.head(2)
test.head(2)
print(train.info())
print(test.info())
sns.pairplot(data=train,vars=['Price','Dep_Time'])
#price outlier check
Q1=train['Price'].quantile(0.25)
Q3=train['Price'].quantile(0.75)
IQR=Q3-Q1

print(Q1)
print(Q3)
print(IQR)
#price outlier removed
train=train[~((train['Price']>Q3+1.5*IQR)|(train['Price']<Q1-1.5*IQR))]
train.shape
train["Duration"] = train["Duration"].astype(int)
test["Duration"] = test["Duration"].astype(int)
train["Journey_Day"] = train["Journey_Day"].astype(object)
test["Journey_Day"] = test["Journey_Day"].astype(object)
train["Journey_Month"] = train["Journey_Month"].astype(object)
test["Journey_Month"] = test["Journey_Month"].astype(object)
train["weekday"] = train["weekday"].astype(object)
test["weekday"] = test["weekday"].astype(object)
import scipy.stats as stats
_ = sns.jointplot(x='Duration',y='Price',data=train,kind = 'reg')
_.annotate(stats.pearsonr)
plt.show()
# Import label encoder 
colnames = list(train.columns)
from sklearn import preprocessing 

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
for col in colnames:
    if train[col].dtype==object:
        train[col]= label_encoder.fit_transform(train[col]) 
train.head()
train.info()
train["weekday"] = train["weekday"].astype(object)
train["Dep_Time"] = train["Dep_Time"].astype(object)
train["Airline"]=train["Airline"].astype(object)
train["Source"]=train["Source"].astype(object)
train["Destination"]=train["Destination"].astype(object)
train["Arrival_Time"]=train["Arrival_Time"].astype(object)
train["Total_Stops"]=train["Total_Stops"].astype(object)
train["Additional_Info"]=train["Additional_Info"].astype(object)
train["Journey_Day"]=train["Journey_Day"].astype(object)
train["Journey_Month"]=train["Journey_Month"].astype(object)
train.head()
train.info()
#for test data
# Import label encoder 
colnames = list(test.columns)
from sklearn import preprocessing 

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
for col in colnames:
    if test[col].dtype==object:
        test[col]= label_encoder.fit_transform(test[col]) 
test.info()
test["weekday"] = test["weekday"].astype(object)
test["Dep_Time"] = test["Dep_Time"].astype(object)
test["Airline"]=test["Airline"].astype(object)
test["Source"]=test["Source"].astype(object)
test["Destination"]=test["Destination"].astype(object)
test["Arrival_Time"]=test["Arrival_Time"].astype(object)
test["Total_Stops"]=test["Total_Stops"].astype(object)
test["Additional_Info"]=test["Additional_Info"].astype(object)
test["Journey_Day"]=test["Journey_Day"].astype(object)
test["Journey_Month"]=test["Journey_Month"].astype(object)
test.info()
test.head()
from sklearn import preprocessing
train["Duration"]= (train["Duration"] - train["Duration"].mean())/train["Duration"].std()   #standardizing
test["Duration"]= (test["Duration"] - test["Duration"].mean())/test["Duration"].std()
X=train.drop(["Price"],axis=1)
Y=train["Price"]
x=np.array(X)
y=np.array(Y)
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
gbm = GradientBoostingRegressor()
xgb = XGBRegressor()
best_gbm = GridSearchCV(gbm, param_grid={'learning_rate':[0.01,0.05,0.1],'max_depth':[1,2,3],'n_estimators':[100,200,500]}, cv=5, n_jobs=-1)
best_xgb = GridSearchCV(xgb, param_grid={'learning_rate':[0.01,0.05,0.1],'max_depth':[1,2,3],'n_estimators':[100,200,500]}, cv=5, n_jobs=-1)
best_gbm.fit(x,y)
best_xgb.fit(x,y)
scores = cross_val_score(best_gbm.best_estimator_, x, y, cv=5)
print("GBM Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
scores = cross_val_score(best_xgb.best_estimator_, x, y, cv=5)
print("XGBoost Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
#KNN
number_of_neighbors = range(1,20)
params = {'n_neighbors':number_of_neighbors}
knn = KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5) 
model.fit(x,y)
scores = cross_val_score(model, x, y, cv=5)
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
#Random Forest
parameters = {'n_estimators':[500], "max_features" : ["auto", "log2", "sqrt"],"bootstrap": [True, False]}
clf = GridSearchCV(RandomForestRegressor(), parameters, n_jobs=-1)
clf.fit(x, y)
scores = cross_val_score(clf, x, y, cv=5)
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
x=np.array(test)
ypred=clf.predict(x)
test= pd.read_excel('../input/flight-fare-prediction-mh/Test_set.xlsx')
test["Price"]=ypred
test.columns
test=test.drop(['Airline', 'Date_of_Journey', 'Source', 'Destination', 'Route',
       'Dep_Time', 'Arrival_Time', 'Duration', 'Total_Stops',
       'Additional_Info'],axis=1)
test.head()
