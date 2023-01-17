import pandas as pd
import numpy as np



# school: okul adı GP-MS
# sex: cinsiyet F-M
# age: yaş
# adress: U-urban(kentsel), R-rural(kırsal)
# famsize: aile boyutu, LE(less then 3,3e eşit veya az ), GT(greater than 3- 3 den büyük)
# pstatus: ebeveyn durumu (A-apart-ayrı), (B-T-together-berber)
# mother's education (numeric: 0 - none, 1 - primary education (4th grade), 
   # 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)
#Fedu: father's education (numeric: 0 - none, 1 - primary education (4th grade), 
   # 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)
#Mjob: mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 
  #('at_home' or 'other') 
# Fjobfather's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 
  # 'at_home' or 'other')
# reasonreason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
# guardianstudent's guardian (nominal: 'mother', 'father' or 'other')
#traveltimehome to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour
# studytime: weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
# failuresnumber of past class failures (numeric: n if 1<=n<3, else 4)- sınıfta kalıp kalmama
#schoolsup: extra educational support (binary: yes or no)
# famsup: family educational support (binary: yes or no)
#paid: extra paid classes within the course subject (Math or Portuguese) (binary: yes or no) extra özel ders
# activities: extra-curricular activities (binary: yes or no)(extra aktiviteler)
# nursery: anaokuluna gidip gitmemek (binary: yes or no)
# higherwants to take higher education (binary: yes or no)
# internet: Internet access at home (binary: yes or no)
# romanticwith a romantic relationship (binary: yes or no)
# famrel: quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
# freetime: free time after school (numeric: from 1 - very low to 5 - very high)
#goout: going out with friends (numeric: from 1 - very low to 5 - very high)
#Dalc: hafta içi alkol kulanımı (numeric: from 1 - very low to 5 - very high)
#Walc:haftasonu alkol kullanımı (numeric: from 1 - very low to 5 - very high)
# health (1 to 5)
# absences:  devmsızlık
# G1: first period grade (numeric: from 0 to 20)
# G2: second period grade (numeric: from 0 to 20)
# G3: final grade (numeric: from 0 to 20)
  
df
df.info()
df.describe().T
df.corr()
import seaborn as sns
sns.pairplot(df,kind = "reg")
df

dms = pd.get_dummies(df[['school', 'sex', 'address','famsize','Pstatus','schoolsup','famsup',
                         'paid','activities','nursery','higher','internet','romantic']])


y= df["G3"]
X_ = df.drop(['Mjob','Fjob','reason','G1','G2','school', 'sex', 'address','famsize','Pstatus','schoolsup',
              'famsup','paid','activities','nursery','higher','internet','romantic','G3'],axis =1)
dms
X = pd.concat([X_, dms[['school_GP', 'sex_F', 'address_R','famsize_GT3','Pstatus_A','schoolsup_no','famsup_no',
                         'paid_no','activities_no','nursery_no','higher_no','internet_no','romantic_no']]], axis=1)
X
X = X.drop(['guardian'],axis=1)
X
y
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
import xgboost as xgb

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25, 
                                                    random_state=42)
knn_model = KNeighborsRegressor().fit(X_train,y_train)
y_pred  =knn_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
knn_model.n_neighbors
knn_params = {'n_neighbors':[4,5,7,10,15]}
knn = KNeighborsRegressor()
knn_cv = GridSearchCV(knn,knn_params, cv=10).fit(X_train,y_train)
knn_cv.best_params_
knn_tuned = KNeighborsRegressor(n_neighbors=10).fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
DM_train = xgb.DMatrix(data= X_train, label=y_train)
DM_test = xgb.DMatrix(data= X_test,label=y_test)
from xgboost import XGBRegressor

xgb_model= XGBRegressor().fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
xgb_model
xgb_grid = {
     'colsample_bytree': [0.4, 0.5,0.6,0.9,1], 
     'n_estimators':[50,100, 200, 500, 1000],
     'max_depth': [2,3,4,5,6],
     'learning_rate': [0.1, 0.01, 0.5]
}
xgb_=XGBRegressor
xgb_cv = GridSearchCV(xgb_model,xgb_grid,cv=10,n_jobs=-1,verbose=2).fit(X_train, y_train)
xgb_cv.best_params_
xgb_tuned = XGBRegressor(colsample_bytree=0.9,
                         learning_rate=0.1,
                         max_depth=2,
                         n_estimators=50).fit(X_train,y_train)
y_pred = xgb_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
