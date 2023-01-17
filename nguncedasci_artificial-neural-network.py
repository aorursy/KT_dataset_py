import pandas as pd
import numpy as np
hitters=pd.read_csv("../input/hittlers/Hitters.csv")
df=hitters.copy()
df=df.dropna()
df.head(2)
y=df["Salary"]
x=df.drop(["League","Division","NewLeague","Salary"],axis=1).astype("float64")
dummies = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
X=pd.concat([x,dummies[["League_N","Division_W","NewLeague_N"]]],axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.25,random_state=42)
#STANDARDIZATION
from sklearn.preprocessing import StandardScaler   #if structures of outliers and variables' variances are so different 
                                                   #than each other, result's confidence decreases so we do standardization.
scaler=StandardScaler()
scaler.fit(X_train)

X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)
from sklearn.neural_network import MLPRegressor
mlp_model=MLPRegressor(hidden_layer_sizes=(100,20)).fit(X_train_scaled,y_train)
mlp_model.n_layers_
mlp_model.hidden_layer_sizes
mlp_model.predict(X_train_scaled)[0:5]
y_pred=mlp_model.predict(X_test_scaled)
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_test,y_pred))
mlp_params={"alpha":[0.01,0.02,0.005],"hidden_layer_sizes":[(20,20),(100,50,150),(300,200,150)], 
            "activation":["relu","logistic"]}

from sklearn.model_selection import GridSearchCV
mlp_cv_model=GridSearchCV(mlp_model,mlp_params,cv=10)
mlp_cv_model.fit(X_train_scaled,y_train)

mlp_cv_model.best_params_
from sklearn.neural_network import MLPRegressor
mlp_tuned=MLPRegressor(alpha= 0.01, hidden_layer_sizes = (100, 50, 150))
mlp_tuned.fit(X_train_scaled,y_train)
y_pred=mlp_tuned.predict(X_test_scaled)
np.sqrt(mean_squared_error(y_test,y_pred))    #THE REAL TEST ERROR
