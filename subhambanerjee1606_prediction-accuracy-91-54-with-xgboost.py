import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency 
from sklearn import preprocessing 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
train=pd.read_excel("../input/flight-fare-prediction-mh/Data_Train.xlsx")
train.head(2)
print(train.isnull().sum())
train=train.dropna()
train=train.drop_duplicates()
#save numeric names
cnames =  ["Price"]
# #Detect and delete outliers from data
for i in cnames:
    q75, q25 = np.percentile(train.loc[:,i], [75 ,25])
    iqr = q75 - q25
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    train = train.drop(train[train.loc[:,i] < min].index)
    train = train.drop(train[train.loc[:,i] > max].index)
train['Journey_Day'] = pd.to_datetime(train.Date_of_Journey, format='%d/%m/%Y').dt.day
train['Journey_Month'] = pd.to_datetime(train.Date_of_Journey, format='%d/%m/%Y').dt.month
train['weekday']= pd.to_datetime(train.Date_of_Journey, format='%d/%m/%Y').dt.weekday
#Transforming duration to minutes
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
#Categorising departure and arrival time
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
train['Arrival_Time']=train['Arrival_Time'].apply(deparrtime)
#Refining total stops column
def stops(x):
    if(x=='non-stop'):
        x=str(0)
    else:
        x.strip()
        stps=x.split(' ')[0]
        x=stps
    return x
train['Total_Stops']=train['Total_Stops'].apply(stops)
train.head(2)
train=train.drop(["Date_of_Journey","Route"],axis=1) #Date of journey, Route dropped because they dont add value to the prediction much
train.head(2)
#Grouping classes with very low frequency with Airline
for i in range(0,len(train["Airline"])):
    if train["Airline"].iloc[i]=="Multiple carriers Premium economy":
         train["Airline"].iloc[i]="Others"
    elif train["Airline"].iloc[i]=="Vistara Premium economy":
         train["Airline"].iloc[i]="Others"
    elif train["Airline"].iloc[i]=="Trujet":
         train["Airline"].iloc[i]="Others"
#Refining No info class in Additional info
for i in range(train.shape[0]):
    if(train['Additional_Info'].iloc[i]=='No info'):
        train['Additional_Info'].iloc[i]='No Info'
#Grouping classes with very low frequency with Additional_Info
for i in range(0,len(train["Additional_Info"])):
    if train["Additional_Info"].iloc[i]== 'No check-in baggage included':
         train["Additional_Info"].iloc[i]="Others"
    elif train["Additional_Info"].iloc[i]=='1 Long layover':
         train["Additional_Info"].iloc[i]="Others"
    elif train["Additional_Info"].iloc[i]== 'Change airports':
         train["Additional_Info"].iloc[i]="Others"
    elif train["Additional_Info"].iloc[i]=='Red-eye flight':
         train["Additional_Info"].iloc[i]="Others"
train.info()
train["Duration"]=train["Duration"].astype(int)
train["weekday"]=train["weekday"].astype(object)
train["Journey_Day"]=train["Journey_Day"].astype(object)
train["Journey_Month"]=train["Journey_Month"].astype(object)
# Import label encoder 
colnames = list(train.columns)
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
for col in colnames:
    if train[col].dtype==object:
        train[col]= label_encoder.fit_transform(train[col])
        train[col]=train[col].astype(object)
train["Duration"]= (train["Duration"] - train["Duration"].mean())/train["Duration"].std()
X=train.drop(["Price"],axis=1)
Y=train["Price"]
x=np.array(X)
y=np.array(Y)
#Ramdom Forest
estimator = RandomForestRegressor()
param_grid = { 
            "n_estimators"      : [100,300,500],
            "max_features"      : ["auto", "sqrt", "log2"],
            "min_samples_split" : [2,4,8],
            "bootstrap": [True,False],
            }
rfmodel = GridSearchCV(estimator, param_grid, n_jobs=-1, cv=5)
rfmodel.fit(x, y)
scores = cross_val_score(rfmodel, x, y, cv=5)
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
#Gradient Boosting and XGBoost
gbm = GradientBoostingRegressor()
xgb = XGBRegressor()
best_gbm = GridSearchCV(gbm, param_grid={'learning_rate':[0.01,0.05,0.1],'max_depth':[3,5,7],'n_estimators':[500]}, cv=5, n_jobs=-1)
best_xgb = GridSearchCV(xgb, param_grid={'learning_rate':[0.01,0.05,0.1],'max_depth':[3,5,7],'n_estimators':[500]}, cv=5, n_jobs=-1)
best_gbm.fit(x,y)
best_xgb.fit(x,y)
scores = cross_val_score(best_gbm.best_estimator_, x, y, cv=5)
print("GBM Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
scores = cross_val_score(best_xgb.best_estimator_, x, y, cv=5)
print("XGBoost Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
#Doing similar pre-processing in test data
test=pd.read_excel("../input/flight-fare-prediction-mh/Test_set.xlsx")
test['Journey_Day'] = pd.to_datetime(test.Date_of_Journey, format='%d/%m/%Y').dt.day
test['Journey_Month'] = pd.to_datetime(test.Date_of_Journey, format='%d/%m/%Y').dt.month
test['weekday']= pd.to_datetime(test.Date_of_Journey, format='%d/%m/%Y').dt.weekday
test['Duration']=test['Duration'].apply(duration)
test['Dep_Time']=test['Dep_Time'].apply(deparrtime)
test['Arrival_Time']=test['Arrival_Time'].apply(deparrtime)
test['Total_Stops']=test['Total_Stops'].apply(stops)
test=test.drop(["Date_of_Journey","Route"],axis=1)
for i in range(0,len(test["Airline"])):
    if test["Airline"].iloc[i]=="Multiple carriers Premium economy":
         test["Airline"].iloc[i]="Others"
    elif test["Airline"].iloc[i]=="Vistara Premium economy":
         test["Airline"].iloc[i]="Others"
    elif test["Airline"].iloc[i]=="Trujet":
         test["Airline"].iloc[i]="Others"
    elif test["Airline"].iloc[i]=="Jet Airways Business":
         test["Airline"].iloc[i]="Others"
for i in range(0,len(test["Additional_Info"])):
    if test["Additional_Info"].iloc[i]== 'No check-in baggage included':
         test["Additional_Info"].iloc[i]="Others"
    elif test["Additional_Info"].iloc[i]=='1 Long layover':
         test["Additional_Info"].iloc[i]="Others"
    elif test["Additional_Info"].iloc[i]== 'Change airports':
         test["Additional_Info"].iloc[i]="Others"
    elif test["Additional_Info"].iloc[i]=='Business class':
         test["Additional_Info"].iloc[i]="Others"
test.info()
test["Duration"]=test["Duration"].astype(int)
test["weekday"]=test["weekday"].astype(object)
test["Journey_Day"]=test["Journey_Day"].astype(object)
test["Journey_Month"]=test["Journey_Month"].astype(object)
# Import label encoder 
colnames = list(test.columns)
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
for col in colnames:
    if test[col].dtype==object:
        test[col]= label_encoder.fit_transform(test[col])
        test[col]=test[col].astype(object)
test["Duration"]= (test["Duration"] - test["Duration"].mean())/test["Duration"].std()
x=np.array(test)
ypred=best_xgb.best_estimator_.predict(x)
test["Price"]=ypred
test["Price"].to_csv("submissionflight.csv")