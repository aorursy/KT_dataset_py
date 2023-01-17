import pandas as pd
import numpy as np
hitters=pd.read_csv("../input/hittlers/Hitters.csv")
df=hitters.copy()
df.head(2)
df=df.dropna()
y=df["Salary"]   # EASY ASSESMENT FOR "SALARY" AND "HITS" VARIABLES
dms=pd.get_dummies(df[["League","Division","NewLeague"]])
dms.head(2)
dms=dms[["League_N","Division_W","NewLeague_N"]]
a=df.drop(["Salary","League","Division","NewLeague"],axis=1)
X=pd.concat([a,dms],axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25, random_state=42)
X_train=pd.DataFrame(X_train["Hits"])
X_test=pd.DataFrame(X_test["Hits"])
from sklearn.svm import SVR
svr_model=SVR("linear").fit(X_train,y_train)
svr_model.predict(X_train)[0:5]
print("y=",svr_model.intercept_[0],"+",svr_model.coef_[0][0],"x") #SVR PREDICTION MODEL
svr_model.coef_[0]
svr_model.coef_[0][0] # add one more square bracket to escape from array
X_train["Hits"][0:1]
-48.69756097561513+4.969512195122206*X_train["Hits"][0:1]   # What will be my salary if i score 91 points in a game? 
                                                            #403
y_pred=svr_model.predict(X_train)
X_train.shape
import matplotlib.pyplot as plt
plt.scatter(X_train,y_train)
plt.plot(X_train,y_pred, color="r");   #SVR GRAPHIC 
#LINEAR REGRESSION FORMULA
from sklearn.linear_model import LinearRegression
linear_model=LinearRegression().fit(X_train,y_train)
linear_predict=linear_model.predict(X_train)
print(linear_model.intercept_, "+",linear_model.coef_[0],"x")
-8.814095480334572+5.1724561354706875*X_train["Hits"][0:1]    # What will be my salary if i score 91 points in a game? 
                                                              #461
#Linear Regression predict is 461, easy SVR model prediction is 403 for scoring 91 points
plt.scatter(X_train,y_train)
plt.plot(X_train,y_pred, color="r")
plt.plot(X_train, linear_model.predict(X_train), color="g")   #Linear Regression(green) and SVR model(red) are together
plt.xlabel("Hits")
plt.ylabel("Salary");
#WHY SVR IS LOWER THAN LINEAR REGRESSION? 
#BECAUSE SVR IS ROBUST, IT IS LESS SENSITIVE TO OUTLIERS THAN LINEAR REGRESSION !!
#IN BRIEFLY, PREDICTIONS ARE:
svr_model.predict([[91]])
linear_model.predict([[91]])
from sklearn.metrics import mean_squared_error    
np.sqrt(mean_squared_error(y_test,svr_model.predict(X_test)))      #test error without tunning
svr_model       # we will tune "c" parameter in this model
svr_params={"C": np.arange(0.1,2,0.1)}
from sklearn.model_selection import GridSearchCV
svr_cv_model=GridSearchCV(svr_model,svr_params,cv=10).fit(X_train,y_train)
svr_cv_model.best_params_
pd.Series(svr_cv_model.best_params_)[0]
svr_tuned_model=SVR("linear",C=pd.Series(svr_cv_model.best_params_)[0]).fit(X_train,y_train)
y_pred=svr_tuned_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
import pandas as pd
from sklearn.model_selection import train_test_split
hitters=pd.read_csv("../input/hittlers/Hitters.csv")
df = hitters.copy()
df = df.dropna()
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df["Salary"]
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25, 
                                                    random_state=42)
svr_params={"C": np.arange(0.1,2,0.1)}
from sklearn.model_selection import GridSearchCV
svr_cv_model=GridSearchCV(svr_model,svr_params,cv=10).fit(X_train,y_train)   #it takes time (more than 15 minutes) 
svr_cv_model
y_pred=svr_cv_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
# BEFORE THIS TIME, WE HAD ONLY ONE X VARIABLE(HITS) AND WE PREDICTED THE SALARY
# NOW, WE HAVE MORE X VARIABLES. 
#AND WE KNOW THAT THESE VARIABLES ALSO HAVE POWER TO EXPLAIN VARIABILITY OF DEPENDENT VARIABLE
#BECAUSE TEST ERROR DECREASED FROM 458 TO 367
#Thanks to https://github.com/mvahit/DSMLBC
