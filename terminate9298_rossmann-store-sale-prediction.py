import os

import pandas as pd

import numpy as np

import scipy

import warnings

warnings.filterwarnings(action='ignore')



# Plotting Library

import seaborn as sns 

import matplotlib.pyplot as plt 

plt.style.use('Solarize_Light2')



# Other Libraries

from sklearn.linear_model import LinearRegression 

from sklearn.metrics import mean_squared_error

from math import sqrt

from scipy.stats import ttest_ind ,linregress , ttest_rel

import statsmodels.api as sm

from scipy.stats import probplot

from scipy.stats import zscore

from sklearn.metrics import r2_score

from statsmodels.graphics.regressionplots import influence_plot

from sklearn.preprocessing import PolynomialFeatures , StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Perceptron

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso ,ElasticNet

from sklearn.ensemble import RandomForestRegressor

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split

print(os.listdir("../input"))
def ispromomonth(rows):

#   if not rows[0].isnull():

    months = {}

    months = str(rows['PromoInterval']).split(',')

    if str(rows['month_str']) in months:

        return 1

    else:

        return 0

def rmspe(y, yhat):

    return np.sqrt(np.mean((yhat/y-1) ** 2))



def rmspe_xg(yhat, y):

    y = np.expm1(y.get_label())

    yhat = np.expm1(yhat)

    return "rmspe", rmspe(y,yhat)



class Rossmann_:

    def __init__(self , train_data_path = '../input/train.csv' , test_data_path='../input/test.csv' , store_path='../input/store.csv' , nrows =100000):

        self.train_data_path =  train_data_path

        self.test_data_path = test_data_path

        self.store_path = store_path

        self.read_size = nrows

        self.train_data_original = pd.read_csv(self.train_data_path , low_memory = False , nrows = self.read_size)

        self.test_data_original = pd.read_csv(self.test_data_path ,low_memory = False , nrows = self.read_size)

        self.store_data_original = pd.read_csv(self.store_path)

        

        self.start_preprocessing_train(self.train_data_original , self.store_data_original)

        self.start_preprocessing_test(self.test_data_original , self.store_data_original)

    

    def start_preprocessing_train(self , train_data , store):

        train_data.StateHoliday = train_data.StateHoliday.replace('0',0)

        train_data.StateHoliday = train_data.StateHoliday.replace('a',1)

        train_data.StateHoliday = train_data.StateHoliday.replace('b',2)

        train_data.StateHoliday = train_data.StateHoliday.replace('c',3)

        train_data['Date_Year'] = train_data['Date'].apply(lambda x: int(x[:4]))

        train_data['Date_Month'] = train_data['Date'].apply(lambda x: int(x[5:7]))

        train_data['Date_Day'] = train_data['Date'].apply(lambda x: int(x[8:]))

        train_data_m = pd.merge(train_data, store, on='Store')

        mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}

        train_data_m.StoreType.replace(mappings, inplace=True)

        train_data_m.Assortment.replace(mappings, inplace=True)

        

        #Finding the week of the year 

        train_data_m['Date'] = pd.to_datetime(train_data_m['Date'], errors='coerce')

        train_data_m['date_WeekOfYear'] = train_data_m.Date.dt.weekofyear

        

        #Combining the Week and Year for Competition and Promo

        train_data_m['Competition_Weeks'] = 12*(train_data_m.Date_Year - train_data_m.CompetitionOpenSinceYear ) + (train_data_m.Date_Month - train_data_m.CompetitionOpenSinceMonth) 

        train_data_m['Promo_Weeks'] = 12*(train_data_m.Date_Year - train_data_m.Promo2SinceYear ) + (train_data_m.Date_Month - train_data_m.Promo2SinceWeek)

        train_data_m['Competition_Weeks'] =  train_data_m['Competition_Weeks'].apply(lambda x: x if x > 0 else 0)

        train_data_m['Promo_Weeks'] =  train_data_m['Promo_Weeks'].apply(lambda x: x if x > 0 else 0)

        

        # is promo month is the months the promo is valid so

        month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}

        train_data_m['month_str'] = train_data_m.Date_Month.map(month2str)

        

        train_data_m['IsPromoMonth'] = train_data_m[[ 'PromoInterval' , 'month_str' ]].apply(ispromomonth , axis = 1) 

        train_data_m.fillna(0, inplace=True)

        

        #updating the rows with sales>0 and customes>0 

        train_data_updated = train_data_m[train_data_m['Sales']>0]

        train_data_updated = train_data_updated[train_data_updated['Customers']>0]

        

        features = ['Store', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo','StateHoliday', 'SchoolHoliday', 'Date_Year', 'Date_Month', 'Date_Day','StoreType', 'Assortment', 'CompetitionDistance','CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2','Promo2SinceWeek', 'Promo2SinceYear', 'date_WeekOfYear', 'Competition_Weeks', 'Promo_Weeks', 'IsPromoMonth']   

                

        self.train_final = train_data_updated[features]

        cols = self.train_final.columns

        self.train_final = pd.DataFrame(StandardScaler().fit_transform(self.train_final) , columns = cols)

        

    def start_preprocessing_test(self , test_data , store):

        test_data.fillna(1 , inplace=True)

        # These are all the Oprations appied on the Data

        test_data['Date_Year'] = test_data['Date'].apply(lambda x: int(x[:4]))

        test_data['Date_Month'] = test_data['Date'].apply(lambda x: int(x[5:7]))

        test_data['Date_Day'] = test_data['Date'].apply(lambda x: int(x[8:]))

        test_data_m = pd.merge(test_data, store, on='Store')

        mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}

        test_data_m.StoreType.replace(mappings, inplace=True)

        test_data_m.Assortment.replace(mappings, inplace=True)

        test_data_m.StateHoliday.replace(mappings, inplace=True)

        test_data_m['Date'] = pd.to_datetime(test_data_m['Date'], errors='coerce')

        test_data_m['date_WeekOfYear'] = test_data_m.Date.dt.weekofyear

        test_data_m['Competition_Weeks'] = 12*(test_data_m.Date_Year - test_data_m.CompetitionOpenSinceYear ) + (test_data_m.Date_Month - test_data_m.CompetitionOpenSinceMonth) 

        test_data_m['Promo_Weeks'] = 12*(test_data_m.Date_Year - test_data_m.Promo2SinceYear ) + (test_data_m.Date_Month - test_data_m.Promo2SinceWeek)

        test_data_m['Competition_Weeks'] =  test_data_m['Competition_Weeks'].apply(lambda x: x if x > 0 else 0)

        test_data_m['Promo_Weeks'] =  test_data_m['Promo_Weeks'].apply(lambda x: x if x > 0 else 0)

        month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}

        test_data_m['month_str'] = test_data_m.Date_Month.map(month2str)

        test_data_m['IsPromoMonth'] = test_data_m[[ 'PromoInterval' , 'month_str' ]].apply(ispromomonth , axis = 1) 

        test_data_m.fillna(0, inplace=True)

        features = ['Store', 'DayOfWeek', 'Open', 'Promo','StateHoliday', 'SchoolHoliday', 'Date_Year', 'Date_Month', 'Date_Day','StoreType', 'Assortment', 'CompetitionDistance','CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2','Promo2SinceWeek', 'Promo2SinceYear', 'date_WeekOfYear', 'Competition_Weeks', 'Promo_Weeks', 'IsPromoMonth']   

        self.test_final  = test_data_m[features]

        

    def prepare_sample_data(self , limit =100 , testing_limit = 30):

        self.data = self.train_final.sample(frac = 1 , random_state = 98).head(limit)

        self.test_data = self.train_final.sample(frac = 1 , random_state = 98).tail(testing_limit)

        

    def Linear_Regression(self):

        print('Creating Linear Regression Model Between Sales and Customers... ')

        lr = LinearRegression()

        lr.fit(self.data['Customers'].values.reshape(-1,1) , self.data['Sales'].values.reshape(-1,1))

        print('Fitting Done on Model ... ')

        print(lr)

        r2_score = lr.score(self.data['Customers'].values.reshape(-1,1), self.data['Sales'].values.reshape(-1,1))

#         print('R2 Score is ',r2_score)

#         print('Since the Model R2 Score is ',r2_score , ', the model explains ',round(r2_score*100,2) , ' % of the variation in GI')

        print('Coefficients for the linear regression problem is ',lr.coef_)

        print('Intersect Value is ',lr.intercept_)

        y_pred = lr.predict(self.data['Customers'].values.reshape(-1, 1))

        rms = sqrt(mean_squared_error(self.data['Sales'].values.reshape(-1,1), y_pred))

        ty_pred = lr.predict(self.test_data['Customers'].values.reshape(-1, 1))

        trms = sqrt(mean_squared_error(self.test_data['Sales'].values.reshape(-1,1), ty_pred))

        print('Root Mean Squared Error of Training Set is ',rms)

        print('Root Mean Squared Error of Testing Set is ',trms)

        

#         print('R2 Score of Training Set is ',r2_score(y_pred, self.data['Sales'].values.reshape(-1,1)))

#         print('R2 Score of Testing Set is ',r2_score(ty_pred, self.test_data['Sales'].values.reshape(-1,1)))

        

        plt.figure(figsize=(15,10))

        plt.scatter(self.data['Customers'].values.reshape(-1, 1) ,  self.data['Sales'].values.reshape(-1,1) , color ='r' , label = 'Actual Values')

        plt.scatter(self.data['Customers'].values.reshape(-1, 1) , y_pred , color ='b' , label = 'Predicted')

        plt.plot(self.data['Customers'].values.reshape(-1, 1) , y_pred , color ='k' , label = 'Predicted Line')

        plt.xlabel('Customers Index')

        plt.ylabel('Sales Index')

        plt.legend()

        plt.savefig('Linear Regression Training.png')

        

        plt.figure(figsize=(15,10))

        plt.scatter(self.test_data['Customers'].values.reshape(-1, 1) ,  self.test_data['Sales'].values.reshape(-1,1) , color ='g' , label = 'Actual Values')

        plt.scatter(self.test_data['Customers'].values.reshape(-1, 1) , ty_pred , color ='y' , label = 'Predicted')

        plt.plot(self.test_data['Customers'].values.reshape(-1, 1) , ty_pred , color ='k' , label = 'Predicted Line')

        plt.xlabel('Customers Index')

        plt.ylabel('Sales Index')

        plt.legend()

        plt.savefig('Linear Regression Testing.png')

        

    def display_graphs(self,simple = False , orders = 1):

        for i in range(1, orders+1,1):

            lm = sns.lmplot(x ="Customers", y ="Sales", data = self.data, scatter = True, order = i, fit_reg = True, ci  = 95 ) 

            lm.fig.suptitle("Scatter plot with Order = "+str(i), fontsize=16)

            

    def Mulitple_Linear_Regression(self):

        print('Creating Multiple Linear Regression Model... ')

        print('Using Columns -> ',self.data.drop(columns = ['Sales','Customers']).columns)

        lr = LinearRegression()

        lr.fit(self.data.drop(columns = ['Sales','Customers']).values , self.data['Sales'].values)

        print(lr)

        print('Fitting Done on Model ... ')

        print('Coefficients for the linear regression problem is ',lr.coef_)

        print('Intersect Value is ',lr.intercept_)

        y_pred = lr.predict(self.data.drop(columns = ['Sales','Customers']).values)

        rms = sqrt(mean_squared_error(self.data['Sales'].values, y_pred))

        ty_pred = lr.predict(self.test_data.drop(columns = ['Sales','Customers']).values)

        trms = sqrt(mean_squared_error(self.test_data['Sales'].values, ty_pred))

        print('Root Mean Squared Error of Training Set is ',rms)

        print('Root Mean Squared Error of Testing Set is ',trms)

#         print('R2 Score of Training Set is ',r2_score(y_pred, self.data['Sales'].values.reshape(-1,1)))

#         print('R2 Score of Testing Set is ',r2_score(ty_pred, self.test_data['Sales'].values.reshape(-1,1)))



        self.data['pred'] = y_pred

        self.test_data['pred'] = ty_pred

        plt.figure(figsize=(15,10))

        sns.jointplot(x = 'Sales' , y = 'pred' , data = self.data, height=10, ratio=3 , color='g' )

        plt.savefig('Multiple Linear Regression Training.png')

        

        plt.figure(figsize=(15,10))

        sns.jointplot(x = 'Sales' , y = 'pred' , data = self.test_data, height=10, ratio=3 , color='r' )

        plt.savefig('Multiple Linear Regression Testing.png')

        

#         plt.figure(figsize=(15,10))

#         plt.scatter(self.test_data['Customers'].values.reshape(-1,1)  ,  self.test_data['Sales'].values.reshape(-1,1) , color ='g',label = 'Actual Values')

#         plt.scatter(self.test_data['Customers'].values.reshape(-1,1)  , ty_pred , color ='y', label = 'Predicted')

#         plt.plot(self.test_data['Customers'].values.reshape(-1,1)  , ty_pred , color ='k' , label = 'Predicted Line')

#         plt.xlabel('Customers Index')

#         plt.ylabel('Sales Index')

#         plt.legend()

#         plt.savefig('Multiple Linear Regression Testing.png')

    

    def Polynomial_Regression(self , degrees = 4):

        print('To Reduce Complexity...\nUsing Single Data Column Customers Rather than All...')

        Input=[('polynomial',PolynomialFeatures(degree=degrees)),('modal',LinearRegression())]

        lr=Pipeline(Input)

        lr.fit(self.data['Customers'].values.reshape(-1,1) , self.data['Sales'].values.reshape(-1,1))

        print('Fitting Done on Model ... ')

        r2_score = lr.score(self.data['Customers'].values.reshape(-1,1), self.data['Sales'].values.reshape(-1,1))

#         print('R2 Score is ',r2_score)

#         print('Since the Model R2 Score is ',r2_score , ', the model explains ',round(r2_score*100,2) , ' % of the variation in GI')

        self.data.sort_values(by='Customers' , inplace = True)

        self.test_data.sort_values(by='Customers' , inplace = True)

        y_pred = lr.predict(self.data['Customers'].values.reshape(-1, 1))

        rms = sqrt(mean_squared_error(self.data['Sales'].values.reshape(-1,1), y_pred))

        ty_pred = lr.predict(self.test_data['Customers'].values.reshape(-1, 1))

        trms = sqrt(mean_squared_error(self.test_data['Sales'].values.reshape(-1,1), ty_pred))

        print('Root Mean Squared Error of Training Set is ',rms)

        print('Root Mean Squared Error of Testing Set is ',trms)

#         print('R2 Score of Training Set is ',r2_score(y_pred, self.data['Sales'].values.reshape(-1,1)))

#         print('R2 Score of Testing Set is ',r2_score(ty_pred, self.test_data['Sales'].values.reshape(-1,1)))



        plt.figure(figsize=(15,10))

        plt.scatter(self.data['Customers'].values.reshape(-1, 1) ,  self.data['Sales'].values.reshape(-1,1) , color ='r',label = 'Actual Values')

        plt.scatter(self.data['Customers'].values.reshape(-1, 1) , y_pred , color ='b', label = 'Predicted')

        plt.plot(self.data['Customers'].values.reshape(-1, 1) , y_pred , color ='k' , label = 'Predicted Line')

        plt.xlabel('Customers Index')

        plt.ylabel('Sales Index')

        plt.legend()

        plt.savefig('Polynomial Regression Training {}.png'.format(degrees))

        

        plt.figure(figsize=(15,10))

        plt.scatter(self.test_data['Customers'].values.reshape(-1, 1) ,  self.test_data['Sales'].values.reshape(-1,1) , color ='g',label = 'Actual Values')

        plt.scatter(self.test_data['Customers'].values.reshape(-1, 1) , ty_pred , color ='y', label = 'Predicted')

        plt.plot(self.test_data['Customers'].values.reshape(-1, 1) , ty_pred , color ='k' , label = 'Predicted Line')

        plt.xlabel('Customers Index')

        plt.ylabel('Sales Index')

        plt.legend()

        plt.savefig('Polynomial Regression Testing {}.png'.format(degrees))

    

    def return_model(self,reg = 'Ridge' , alpha = 0.01):

        if reg == 'Ridge':

            lr = Ridge(alpha=alpha)

        elif reg =='Lasso':

            lr = Lasso(alpha=alpha)

        elif reg =='Elastic':

            lr = ElasticNet(alpha = alpha)

        else:

            lr = Ridge(alpha=alpha , solver = 'cholesky', tol = .005)

        return lr

    

    def Other_Regression(self , reg = 'Ridge'):

        print('Creating Multiple {} Regression Model... '.format(reg))

        print('Using Columns -> ',self.data.drop(columns = ['Sales','Customers']).columns)

        lr = self.return_model(reg = reg)

        lr.fit(self.data.drop(columns = ['Sales','Customers']).values , self.data['Sales'].values)

        print(lr)

        print('Fitting Done on Model ... ')

        print('Coefficients for the linear regression problem is ',lr.coef_)

        print('Intersect Value is ',lr.intercept_)

        y_pred = lr.predict(self.data.drop(columns = ['Sales','Customers']).values)

        rms = sqrt(mean_squared_error(self.data['Sales'].values, y_pred))

        ty_pred = lr.predict(self.test_data.drop(columns = ['Sales','Customers']).values)

        trms = sqrt(mean_squared_error(self.test_data['Sales'].values, ty_pred))

        print('Root Mean Squared Error of Training Set is ',rms)

        print('Root Mean Squared Error of Testing Set is ',trms)

        

        print('Creating Alpha VS Mean Squared Error Graph for Alpha')

        alphas = []

        train_loss = []

        test_loss = []

        for i in range(10000):

            alphas.append(i*0.0015 +0.0001)

            lr = self.return_model(reg = reg , alpha = (i*0.0015 +0.0001))

            lr.fit(self.data.drop(columns = ['Sales','Customers']).values , self.data['Sales'].values)

            y_pred = lr.predict(self.data.drop(columns = ['Sales','Customers']).values)

            rms = sqrt(mean_squared_error(self.data['Sales'].values, y_pred))

            ty_pred = lr.predict(self.test_data.drop(columns = ['Sales','Customers']).values)

            trms = sqrt(mean_squared_error(self.test_data['Sales'].values, ty_pred))

            train_loss.append(rms)

            test_loss.append(trms)

        

        plt.figure(figsize=(15,10))

        plt.plot(alphas , train_loss , color ='r' , label = 'Training Loss')

        plt.xlabel('Alpha')

        plt.ylabel('Loss (RMSE)')

        plt.legend()

        plt.savefig('{} Regression Alpha Training.png'.format(reg))

        plt.figure(figsize=(15,10))

        plt.plot(alphas , test_loss , color ='g' , label = 'Testing Loss')

        plt.xlabel('Alpha')

        plt.ylabel('Loss (RMSE)')

        plt.legend()

        plt.savefig('{} Regression Alpha Testing.png'.format(reg))

        

        print('Using Single Column now ....')

        lr = self.return_model(reg = reg)

        

        lr.fit(self.data['Customers'].values.reshape(-1,1) , self.data['Sales'].values.reshape(-1,1))

        print('Fitting Done on Model ... ')

        print(lr)

        r2_score = lr.score(self.data['Customers'].values.reshape(-1,1), self.data['Sales'].values.reshape(-1,1))

        print('Coefficients for the linear regression problem is ',lr.coef_)

        print('Intersect Value is ',lr.intercept_)

        y_pred = lr.predict(self.data['Customers'].values.reshape(-1, 1))

        rms = sqrt(mean_squared_error(self.data['Sales'].values.reshape(-1,1), y_pred))

        ty_pred = lr.predict(self.test_data['Customers'].values.reshape(-1, 1))

        trms = sqrt(mean_squared_error(self.test_data['Sales'].values.reshape(-1,1), ty_pred))

        print('Root Mean Squared Error of Training Set is ',rms)

        print('Root Mean Squared Error of Testing Set is ',trms)

        

        plt.figure(figsize=(15,10))

        plt.scatter(self.data['Customers'].values.reshape(-1, 1) ,  self.data['Sales'].values.reshape(-1,1) , color ='r' , label = 'Actual Values')

        plt.scatter(self.data['Customers'].values.reshape(-1, 1) , y_pred , color ='b' , label = 'Predicted')

        plt.plot(self.data['Customers'].values.reshape(-1, 1) , y_pred , color ='k' , label = 'Predicted Line')

        plt.xlabel('Customers Index')

        plt.ylabel('Sales Index')

        plt.legend()

        plt.savefig('{} Regression Training.png'.format(reg))

        

        plt.figure(figsize=(15,10))

        plt.scatter(self.test_data['Customers'].values.reshape(-1, 1) ,  self.test_data['Sales'].values.reshape(-1,1) , color ='g' , label = 'Actual Values')

        plt.scatter(self.test_data['Customers'].values.reshape(-1, 1) , ty_pred , color ='y' , label = 'Predicted')

        plt.plot(self.test_data['Customers'].values.reshape(-1, 1) , ty_pred , color ='k' , label = 'Predicted Line')

        plt.xlabel('Customers Index')

        plt.ylabel('Sales Index')

        plt.legend()

        plt.savefig('{} Regression Testing.png'.format(reg))
ross = Rossmann_()
ross.prepare_sample_data(limit =200 , testing_limit = 40)

ross.Linear_Regression()
ross.prepare_sample_data(limit =2000 , testing_limit = 400)

ross.Mulitple_Linear_Regression()
ross.prepare_sample_data(limit =10000 , testing_limit = 4000)

ross.Polynomial_Regression(degrees = 3)
ross.prepare_sample_data(limit =10000 , testing_limit = 4000)

ross.Polynomial_Regression(degrees = 2)
ross.prepare_sample_data(limit =1000 , testing_limit = 400)

ross.Other_Regression(reg = 'Ridge')
ross.prepare_sample_data(limit =1000 , testing_limit = 400)

ross.Other_Regression(reg = 'Lasso')
ross.prepare_sample_data(limit =1000 , testing_limit = 400)

ross.Other_Regression(reg = 'Elastic')
ross.prepare_sample_data(limit =1000 , testing_limit = 400)

ross.Other_Regression(reg = 'Bridge')
# now using Xgb 