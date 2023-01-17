import os
import pandas as pd
import pandas_profiling
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
data = pd.read_csv('../input/melbourne-housing-market/MELBOURNE_HOUSE_PRICES_LESS.csv')
data.head()
missing_val_count_by_column = (data.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
data1 = data.dropna(subset=['Price'])
(data1.isnull().sum())
data1.describe(percentiles = [.01,.99])
data1.loc[data1['Rooms'] == 31]
data1.loc[data1['Price'] == 1.120000e+07]
data1.info()
data1.describe(include=['O'])
profile = pandas_profiling.ProfileReport(data1)
profile.to_widgets()
import re
def to_street(str):
    return re.sub('[^A-Za-z]+', '', str)
data1.Address.apply(to_street).value_counts().count()
data2 = data1.drop(columns=['Address', 'Suburb', 'CouncilArea'])
import datetime
def to_year(date_str):
    return datetime.datetime.strptime(date_str.strip(),'%d/%m/%Y').year
data2['Date'] = data2.Date.apply(to_year)
data2.Date.value_counts()
counts = data2.SellerG.value_counts()
data2.SellerG[data2['SellerG'].isin(counts[counts < 100].index)] = 'less than 100'
data2.SellerG[data2['SellerG'].isin(counts[(counts >= 100) & (counts < 200)].index)] = '100 - 200'
data2.SellerG[data2['SellerG'].isin(counts[(counts >= 200) & (counts < 500)].index)] = '200 - 500'
data2.SellerG[data2['SellerG'].isin(counts[(counts >= 500) & (counts < 1000)].index)] = '500 - 1000'
data2.SellerG[data2['SellerG'].isin(counts[counts > 1000].index)] = 'over 1000'
data2.SellerG.value_counts()
data2.head()
data2.describe(include=['O']).T
data3 = pd.get_dummies(data2)
data3.head()
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error
train, test = train_test_split(data3, test_size = 0.2, random_state=512)
print ('Train:', train.shape)
print ('Test:', test.shape)
X_train = train.loc[:, data3.columns != 'Price']
y_train = train.Price

X_test = test.loc[:, data3.columns != 'Price']
y_test = test.Price
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import ResidualsPlot
import matplotlib.pyplot as plt
import seaborn as sns


A = Ridge(alpha=0)
A.fit(X_train, y_train)
visualizer = ResidualsPlot(A)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()   

model_CV = linear_model.LinearRegression()
from sklearn.model_selection import cross_val_score
import plotly.express as px

B_score = []
cv = []

for i in range(2, 11):
    model = Ridge(alpha=0, normalize=True)
    score = cross_val_score(model, X_train, y_train, cv=i).mean()
    B_score.append(score)
    cv.append(i)
    
    print("cv: %d --- score: %2.4f" % (i, score))
    
px.line(x=cv, y=B_score, 
        template='simple_white', 
        title='<b>K-fold vs R2</b>',
        labels={'x':'K-fold', 'y':'R2'})
from sklearn.model_selection import GridSearchCV

params = {'alpha':[100, 30, 21, 20, 19.5, 19, 18.5, 18, 17, 17.5, 16, 15, 14, 13.5, 13, 12.5, 12, 11, 10.5, 10, 9.5, 9, 8.5, 8, 7.7, 7.6, 7.5, 7.4, 7.3, 7, 6, 5, 4.5, 4, 3.5, 3, 1, 0.3, 0.1, 0.03, 0.01, 0],
          'normalize': (True, False)}

model = Ridge()
gsc = GridSearchCV(estimator=model, param_grid=params, cv = 10, n_jobs=-1)
gsc.fit(X_train, y_train)

best = gsc.best_params_
score = gsc.score(X_test, y_test)
print('With : ', best)
print('Score: %2.4f' % score)
B = gsc.best_estimator_
B.fit(X_train, y_train)
print("B's score: %2.4f" % B.score(X_test, y_test))

visualizer = ResidualsPlot(B)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show() 
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
pca = PCA()
pca.fit(X_train)
C = Pipeline([
                ('PCA', PCA(n_components=10)),
                ('Linear Regression', Ridge(alpha=0, normalize=True))])
C.fit(X_train, y_train)
C.score(X_test, y_test)

visualizer = ResidualsPlot(C)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show() 
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV

step = [( 'PCA'     , PCA()   ),
        ( 'Lin_Reg' , RidgeCV(alphas=[0], cv=10) )]

D = Pipeline(step)
D.fit(X_train, y_train)
score = D.score(X_test, y_test)
print("D's score: %2.4f" % score)
step = [( 'PCA'     , PCA()   ),
        ( 'Lin_Reg' , Ridge() )]
pipe = Pipeline(step)

params = {'PCA__n_components' : range(1,24),
          'Lin_Reg__alpha'    : [100, 30, 21, 20, 19.5, 19, 18.5, 18, 17, 17.5, 16, 15, 14, 13.5, 13, 12.5, 12, 11, 10.5, 10, 9.5, 9, 8.5, 8, 7.7, 7.6, 7.5, 7.4, 7.3, 7, 6, 5, 4.5, 4, 3.5, 3, 1, 0.3, 0.1, 0.03, 0.01, 0],
          'Lin_Reg__normalize': [True, False]}

gsc = GridSearchCV(pipe, param_grid=params, cv=7)
gsc.fit(X_train, y_train)

best = gsc.best_params_
score = gsc.score(X_test, y_test)
print('With : ', best)
print('Score: %2.4f' % score)
D = gsc.best_estimator_
D.fit(X_train, y_train)
print("D's score: %2.4f" % D.score(X_test, y_test))

visualizer = ResidualsPlot(D)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show() 