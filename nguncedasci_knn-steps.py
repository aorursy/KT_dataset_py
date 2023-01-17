import pandas as pd
import numpy as np
hitters=pd.read_csv("../input/hittlers/Hitters.csv")
df=hitters.copy()
df.columns

df.head(2)
df.info()
df.shape
import pandas as pd
import numpy as np
df=df.dropna()
df.shape
dms=pd.get_dummies(df[['League', 'Division', 'NewLeague']])  
dms.head(2)
x=df.drop(["Salary",'League', 'Division', 'NewLeague'], axis=1).astype("float64")
x.head(2)

X = pd.concat([x, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
X.head(2)
y=df["Salary"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test= train_test_split(X,y,test_size=0.25, random_state=42)

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

knn=KNeighborsRegressor()
k_params={"n_neighbors":np.arange(1,30,1)}


knncv=GridSearchCV(knn,k_params, cv=10)
knncv.fit(X_train, y_train)

knncv.best_params_["n_neighbors"]   #the best k for this dataset
from sklearn.metrics import mean_squared_error, r2_score 
knn_tune=KNeighborsRegressor(n_neighbors = knncv.best_params_["n_neighbors"]).fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, knn_tune.predict(X_test)))
# Create KNN model object with default settings (n_neighbors=5 which means k value=5)
from sklearn.neighbors import KNeighborsRegressor
knn_model=KNeighborsRegressor().fit(X_train, y_train)
knn_model

knn_model.n_neighbors
#Find Test Error (RMSE) within predicted y values and "test data real y values". 
#To do that, first predict y values from "test data x values". 
#In brief,from test data x values--> gain predicted y values and compare it with test data real y values
ypredict=knn_model.predict(X_test)
ypredict[0:5]
#### Find Test Error (RMSE) within predicted y values and "test data y values.
from sklearn.metrics import mean_squared_error, r2_score 
np.sqrt(mean_squared_error(ypredict,y_test))
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
RMSE=[]     
RMSECV=[]    #rmse values with cross validation.
for k in range(10):
    k=k+1
    knn_model = KNeighborsRegressor(n_neighbors = k).fit(X_train, y_train)    #train error
    ytrainpredict=knn_model.predict(X_train)
    rmse=np.sqrt(mean_squared_error(ytrainpredict,y_train))
    rmsecv=np.sqrt(-1*cross_val_score(knn_model,X_train,y_train,cv=10, scoring="neg_mean_squared_error").mean())
    RMSE.append(rmse)
    RMSECV.append(rmsecv)
    print("for k=", k, "    RMSE value=",rmse, "     Cross Validated RMSE", rmsecv)
    #Here the result of RMSE with CV : the range of it is btw 283 and 301. It is smaller than rmse range.
knn_tune=KNeighborsRegressor(n_neighbors = knncv.best_params_["n_neighbors"]).fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, knn_tune.predict(X_test)))
