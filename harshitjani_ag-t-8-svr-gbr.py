import pandas as pd

avgmdata= pd.read_csv("../input/dataset6/Ag_t8.csv")
avgmdata=avgmdata[0:14690]
avgmdata
avgmdata.isna().sum()
avgmdata.dropna(0,inplace=True)
avgmdata.columns
X = avgmdata[["Time",'windgust','pressure','temperature','winddegree','windspeed',"AG",'t1','t2','t3','t4','t5',"t6","t7","t8",'t9','t10']]

y = avgmdata[['target']]
X
y
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.30, random_state=101)

from sklearn.svm import SVR
model = SVR(kernel = 'rbf',C=1000,gamma=0.001,epsilon=5)
model.fit(x_train,y_train)
ypred = clf.predict(x_test)
predictions = pd.DataFrame(ypred)

actual = pd.DataFrame(y_test)
actual=actual.reset_index()
compare = pd.concat([predictions,actual],axis=1)
compare.to_csv('comparexgb.csv')
print(y_test)
from sklearn.metrics import mean_squared_error

mean_squared_error(ypred,y_test)
t=[[20,32,1014,23,250,16,90.74,81.07,75.63,82.13,70.82,66.53,63.36,58.43,50.52,44.78,35]]

model.predict(t)




from sklearn import ensemble

from sklearn import datasets

from sklearn.utils import shuffle

from sklearn.metrics import mean_squared_error

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss': 'ls'}

clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(x_test, y_test)



t=[[20,32,1014,23,250,16,90.74,81.07,75.63,82.13,70.82,66.53,63.36,58.43,50.52,44.78,35]]

clf.predict(t)