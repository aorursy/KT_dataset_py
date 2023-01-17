import pandas as pd 

import numpy as np 

from sklearn.model_selection import train_test_split 

  

# import the dataset 

train_df = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv') 

test_df  = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')



train_df = train_df.fillna(0.00)

test_df = test_df.fillna(0.00)  



x_train = train_df[['feature1','feature2','feature3','feature5','feature6','feature7','feature9','feature10']] 

y_train = train_df['rating'] 

x_test  = test_df[['feature1','feature2','feature3','feature5','feature6','feature7','feature9','feature10']]



'''

lm = LinearRegression()

lm.fit(x_train,y_train)

pred = lm.predict(x_test)



rr = Ridge(alpha = 0.0009)

rr.fit(x_train,y_train)

pred = rr.predict(x_test)



las = Lasso(alpha = 0.01)

las.fit(x_train,y_train)

pred = las.predict(x_test)



clf = SVC(kernel='linear') 

clf.fit(x_train,y_train)

pred = clf.predict(x_test)



elastic_net_cv = ElasticNetCV(normalize=True, alphas=np.logspace(-10, 1, 400), 

                              l1_ratio=np.linspace(0, 1, 100))

elastic_net_model = elastic_net_cv.fit(x_train, y_train)

pred = elastic_net_model.predict(x_test)





elastic=ElasticNet(normalize=True)

search=GridSearchCV(estimator=elastic,param_grid={'alpha':np.logspace(-5,2,8),'l1_ratio':[.2,.4,.6,.8]},scoring='neg_mean_squared_error',n_jobs=1,refit=True,cv=10)

search.fit(x_train,y_train)

pred = search.predict(x_test)

'''



import csv

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.svm import SVC

from sklearn.linear_model import ElasticNetCV

from sklearn.linear_model import ElasticNet

from sklearn.datasets import make_regression

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor 



regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 

regressor.fit(x_train,y_train)

pred = regressor.predict(x_test)





#typecasting double to int 

for i in range(len(pred)):

    pred[i] =int(round(pred[i]))



pred = pred.astype(int)

fields = ['id','rating']

rows = zip(test_df['id'],pred)



filename = 'sample.csv'

with open(filename, 'w') as csvfile: 

    csvwriter = csv.writer(csvfile) 

    csvwriter.writerow(fields) 

    csvwriter.writerows(rows)



trail = pd.read_csv('sample.csv')

modified = trail.dropna()

modified.to_csv('sample.csv',index=False)
import pandas as pd 

import numpy as np 

from sklearn.model_selection import train_test_split 

  

# import the dataset 

train_df = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv') 

test_df  = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')



train_df = train_df.fillna(0.00)

test_df = test_df.fillna(0.00)  



x_train = train_df[['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11']] 

y_train = train_df['rating'] 

x_test  = test_df[['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11']]



'''

lm = LinearRegression()

lm.fit(x_train,y_train)

pred = lm.predict(x_test)



rr = Ridge(alpha = 0.0009)

rr.fit(x_train,y_train)

pred = rr.predict(x_test)



las = Lasso(alpha = 0.01)

las.fit(x_train,y_train)

pred = las.predict(x_test)



clf = SVC(kernel='linear') 

clf.fit(x_train,y_train)

pred = clf.predict(x_test)



elastic_net_cv = ElasticNetCV(normalize=True, alphas=np.logspace(-10, 1, 400), 

                              l1_ratio=np.linspace(0, 1, 100))

elastic_net_model = elastic_net_cv.fit(x_train, y_train)

pred = elastic_net_model.predict(x_test)





elastic=ElasticNet(normalize=True)

search=GridSearchCV(estimator=elastic,param_grid={'alpha':np.logspace(-5,2,8),'l1_ratio':[.2,.4,.6,.8]},scoring='neg_mean_squared_error',n_jobs=1,refit=True,cv=10)

search.fit(x_train,y_train)

pred = search.predict(x_test)

'''



import csv

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.svm import SVC

from sklearn.linear_model import ElasticNetCV

from sklearn.linear_model import ElasticNet

from sklearn.datasets import make_regression

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor 



regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 

regressor.fit(x_train,y_train)

pred = regressor.predict(x_test)





#typecasting double to int 

for i in range(len(pred)):

    pred[i] =int(round(pred[i]))



pred = pred.astype(int)

fields = ['id','rating']

rows = zip(test_df['id'],pred)



filename = 'sample.csv'

with open(filename, 'w') as csvfile: 

    csvwriter = csv.writer(csvfile) 

    csvwriter.writerow(fields) 

    csvwriter.writerows(rows)



trail = pd.read_csv('sample.csv')

modified = trail.dropna()

modified.to_csv('sample.csv',index=False)