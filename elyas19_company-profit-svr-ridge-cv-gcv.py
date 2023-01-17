import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd
data = pd.read_csv('/kaggle/input/50_Startups.csv')

data.info()

data.head(5)
corr_matrix = data.corr()

sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",fmt=".2f",annot_kws={'size':16})

corr_matrix['Profit']
sns.set(style="ticks")

sns.pairplot(data)

plt.show()
Y = data.iloc[:,-1].values.reshape((len(data),1))

X = data.iloc[:,0].values.reshape((len(data),1))
plt.figure(figsize=(10,5))

plt.scatter(X,Y,label='data')

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge



poly_feature = PolynomialFeatures(degree=10)

X_poly = poly_feature.fit_transform(X)

#creat input data for pltting 

plot_x = np.arange(0,180000,1000).reshape((180,1))

plot_X_poly = poly_feature.fit_transform(plot_x)



linreg = LinearRegression()

linreg.fit(X_poly,Y)

plt.plot(plot_x,linreg.predict(plot_X_poly),'r',label='linear regressor')



ridgreg = Ridge(alpha=1,solver='cholesky')

ridgreg.fit(X_poly,Y)

#plt.scatter(X,ridgreg.predict(X_poly))

plt.plot(plot_x,ridgreg.predict(plot_X_poly),'k',label='Ridge regressor')

plt.xlim([min(X)-5000,max(X)+5000])

plt.ylim([min(Y)-5000,max(Y)+5000])

plt.xlabel('R&D Spend');plt.ylabel('Profit');plt.legend()

plt.show()
from sklearn.model_selection import cross_validate

cv_results = cross_validate(ridgreg,X_poly,Y,cv=5,scoring="neg_mean_squared_error")

for a,b in zip(np.sqrt(-cv_results['train_score']),np.sqrt(-cv_results['test_score'])):

    print("training error:",a,"testing error:",b)

print("average training error:",np.mean(np.sqrt(-cv_results['train_score'])),

      "average testing error:",np.mean(np.sqrt(-cv_results['test_score'])))



plt.scatter([1,2,3,4,5],np.log(np.sqrt(-cv_results['train_score'])),label='train RMSE')

plt.scatter([1,2,3,4,5],np.log(np.sqrt(-cv_results['test_score'])),label = 'test RMSE')

plt.xlabel('fold number');plt.ylabel('log RMSE');plt.legend();plt.show()
plt.scatter(X,Y,label='data')

from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler



def test_regressor(svm_reg):

    # preprocessing data , otherwise wont coverge with SVR

    scx = StandardScaler();X_sc = scx.fit_transform(X)

    scy = StandardScaler();Y_sc = scy.fit_transform(Y)



    svm_reg.fit(X_sc,Y_sc.ravel())

    #creat input data for pltting 

    plot_x = np.arange(0,180000,1000).reshape((180,1))

    y_sc_plot_pre = svm_reg.predict(scx.fit_transform(plot_x))

    y_plot_pre = scy.inverse_transform(y_sc_plot_pre)

    plt.plot(plot_x,y_plot_pre,c='k',label='SVR regresoor')

    plt.xlabel('R&D Spend');plt.ylabel('Profit');plt.legend()

    plt.show()

    return X_sc,Y_sc

    #y_sc_predict = svm_poly_reg.predict(X_sc)

    #y_predict = scy.inverse_transform(y_sc_predict)

    #plt.plot(X,y_predict)



X_sc,Y_sc = test_regressor(SVR(kernel='poly',degree=3,C=100,gamma=0.1,epsilon=0.001))
from sklearn.model_selection import GridSearchCV

param_grid = {

    'C':[10,100,1000],

    'degree':[1,3,5],

    'epsilon':[0.01,0.1]

}

svm_poly_reg = SVR(kernel='poly',gamma='auto')

grid_search = GridSearchCV(svm_poly_reg,param_grid,cv=5,scoring='neg_mean_squared_error')

grid_search.fit(X_sc,Y_sc.ravel())
gcv_res =grid_search.cv_results_

for mean_score,params in zip(gcv_res["mean_test_score"],gcv_res["params"]):

    print(np.sqrt(-mean_score),params)
plt.scatter(X,Y,label='data')

test_regressor(grid_search.best_estimator_)