## import library

!pip install quandl

import warnings

warnings.filterwarnings("ignore") 

import pandas as pd   

import quandl

import numpy as np

import matplotlib.pyplot as plt  #for plotting

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn import preprocessing

#from model_selection import cross_validation

from sklearn.svm import SVR

#from mlxtend.regressor import StackingRegressor

from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error



## data source 



df=quandl.get('WIKI/GOOGL')
##data summary

df.head()
## redefining data adding removin feture

### create the specfic ammount of label and feture 

df1=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]



###redefining the data

### adding some feture to the datasets

df1['volatility']=(df1['Adj. High']-df1['Adj. Close'])/df1['Adj. Close']

df1['PCT_Change']=(df1['Adj. Close']-df1['Adj. Open'])/df1['Adj. Open'] 
## making final dataframe

df1=df1[['Adj. Close','volatility','PCT_Change','Adj. Open','Adj. Volume']]
## setting the target column

forcast_col='Adj. Close'
## deal with the null data

df1.fillna(-999999,inplace=True)
## for predicting one percent of the data

import math

forcast_out = int(math.ceil(.1*(len(df1))))

print (forcast_out)
## displaying the previous output

Y=df1[forcast_col]

X=range(len(df1[forcast_col]))

fig_size=[30,5]

plt.rcParams["figure.figsize"] = fig_size

plt.plot(X,Y)
##storing the previous data in  a dataframe

df1['label'] = df[forcast_col].shift(-forcast_out)

y1 = df1['label']

x1=range(len(df1['label']))

fig_size=[30,5]

plt.rcParams["figure.figsize"] = fig_size

plt.plot(x1,y1)
## dropping the first column which is the output

X=np.array(df1.drop(['label'],1))
##scale the data

X=preprocessing.scale(X)

X=X[:-forcast_out]  ##data what is known

X_lately=X[-forcast_out:] ##data we predict

df1.dropna(inplace=True)
Y=np.array(df1['label'])
##split the training and testing data

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2)
## training separtely the classifier

##first knn

n_neighbors=1

clf1 = KNeighborsRegressor(n_neighbors)  # create a classifire object

clf1.fit(xtrain,ytrain) # train data related with fir() method

accuracy1=clf1.score(xtest,ytest) # test data related with score() method

print ("the accuracy is "+str(accuracy1))
## second linear regression

from sklearn.linear_model import LinearRegression

clf2 = LinearRegression()  # create a classifire object

clf2.fit(xtrain,ytrain) # train data related with fir() method

accuracy2=clf2.score(xtest,ytest) # test data related with score() method

print ("the accuracy is "+str(accuracy2))

## third support vector machine

from sklearn import svm

clf3 = svm.SVR()  # create a classifire object

clf3.fit(xtrain,ytrain) # train data related with fir() method

accuracy3=clf3.score(xtest,ytest) # test data related with score() method

print ("the accuracy is "+str(accuracy3))
clf4 = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=100)

clf4.fit(xtrain,ytrain) # train data related with fir() method

accuracy4=clf4.score(xtest,ytest) # test data related with score() method

print ("the accuracy is "+str(accuracy4))
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)   

averaged_models = AveragingModels(models = (clf1, clf2, clf3, clf4))
averaged_models.fit(xtrain,ytrain)
accuracy=averaged_models.score(xtest,ytest)
accuracy


df2=pd.DataFrame()

df3=pd.DataFrame()

df4=pd.DataFrame()

df5=pd.DataFrame()

df6=pd.DataFrame()
forcast_set1=clf1.predict(X_lately)

forcast_set2=clf2.predict(X_lately)

forcast_set3=clf3.predict(X_lately)

forcast_set4=clf4.predict(X_lately)

final_forcast_set=averaged_models.predict(X_lately)

df2['forcast']=np.array(forcast_set1)

df3['forcast']=np.array(forcast_set2)

df4['forcast']=np.array(forcast_set3)

df5['forcast']=np.array(forcast_set4)

df6['forcast']=np.array(final_forcast_set)

fig_size=[30,30]

plt.rcParams["figure.figsize"] = fig_size

df2['forcast'].plot()

df3['forcast'].plot()

df4['forcast'].plot()

df5['forcast'].plot()

df6['forcast'].plot()

plt.legend(loc=4)



plt.ylabel('Price')