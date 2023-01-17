# Basic Libraries

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

from statsmodels.graphics.regressionplots import influence_plot
class country:

    def __init__(self , add = '/kaggle/input/Machine Learning (Codes and Data Files)/Data/country.csv'):

        self.add = add

        self.df = pd.read_csv(add)

    def print_details(self , df = None):

        if df is None:

            df = self.df

        print('Shape of file is ',df.shape)

        print('Description of file is ........')

        display(df.describe())

        print('Info of file is ......')

        display(df.info())

        print('Random 5 Sample from file is .......')

        display(df.sample(5))

        

    def display_corr(self):

        ax = sns.heatmap(self.df[['Corruption_Index' , 'Gini_Index']].corr() , annot=True)

        

    def display_graphs(self,simple = False , orders = 1):

        for i in range(1, orders+1):

#             plt.subplot(2, 3, i)

            lm = sns.lmplot(x ="Corruption_Index", y ="Gini_Index", data = self.df, scatter = True, order = i, fit_reg = True, ci  = 95 ) 

            lm.fig.suptitle("Scatter plot with Order = "+str(i), fontsize=16)

        

    def get_standardized_values( self,vals ):

        return (vals - vals.mean())/vals.std()

    

    def Question_1(self):

        # Designing simple Linear Regression Model

        print('Creating Linear Regression Model ... ')

        lr = LinearRegression()

        lr.fit(self.df['Gini_Index'].values.reshape(-1, 1) , self.df['Corruption_Index'].values)

        print('Fitting Done on Model ... ')

        self.r2_score = lr.score(self.df['Gini_Index'].values.reshape(-1, 1) , self.df['Corruption_Index'])

        print('R2 Score is ',self.r2_score)

        print('Since the Model R2 Score is ',self.r2_score , ', the model explains ',round(self.r2_score*100,2) , ' % of the variation in GI')

        print('Coefficients for the linear regression problem is ',lr.coef_)

        print('Intersect Value is ',lr.intercept_)

        y_pred = lr.predict(self.df['Gini_Index'].values.reshape(-1, 1))

        self.rms = sqrt(mean_squared_error(self.df['Corruption_Index'].values.reshape(-1,1), y_pred))

        print('Root Mean Squared Is ',self.rms)

        plt.scatter(self.df['Gini_Index'].values.reshape(-1, 1) , y_pred , color ='b')

        plt.scatter(self.df['Gini_Index'].values.reshape(-1, 1) ,  self.df['Corruption_Index'].values.reshape(-1,1) , color ='r')

        plt.plot(self.df['Gini_Index'].values.reshape(-1, 1) , y_pred , color ='k') 

        plt.xlabel('Gini Index')

        plt.ylabel('Corruption Index')

        plt.show() 

        

    def fn_2(self):

        self.linreg = linregress(self.df['Gini_Index'],self.df['Corruption_Index'])

        self.sm =sm.OLS(self.df['Corruption_Index'], self.df['Gini_Index'] ).fit()

#         print(self.sm.summary2())

#         print(self.sm.conf_int(0.95) )



    def fn_3(self):

        print('Two-sided test for the null hypothesis that 2 related \n or repeated samples have identical average (expected) values is')

        print(ttest_rel(self.df['Gini_Index'].values,self.df['Corruption_Index'].values ))

        print('Two-sided test for the null hypothesis that 2 independent \n samples have identical average (expected) values is ')

        print(ttest_ind(self.df['Gini_Index'].values,self.df['Corruption_Index'].values ))

#         print('Since p-value for the t-test is 2.71346767e-06 which is lesser than .1 .\nTherefore Fail to Reject H0 Hypothesis.')

    def Question_3(self):

        print('Two-sided p-value for a hypothesis test whose null hypothesis \nis that the slope is zero, using Wald Test with t-distribution \nof the test statistic is \n',self.linreg[3])

    def fn_4(self):

        self.corruption_resid = self.sm.resid

        probplot(self.corruption_resid, plot=plt)

    

    def fn_5(self):

        plt.scatter( self.get_standardized_values( self.sm.fittedvalues ),self.get_standardized_values( self.corruption_resid ))

        

    def fn_6(self):

        self.slope = self.linreg[0]

        self.se = self.linreg[4]

        print('The Confidence Intervel for Regression Coefficient b1 is ')

        print('The Slope of Linear Regression is ', self.slope)

        print('t Value for n-k-1 at 95% Confidence is ',scipy.stats.t.ppf(0.05/2.,18) )

        print('Upper Bound is ' ,self.slope - self.se* scipy.stats.t.ppf(0.05/2.,18))

        print('Lower Bound is ' ,self.slope + self.se* scipy.stats.t.ppf(0.05/2.,18))

    

    def fn_7(self):

        self.zscore = zscore(count.df['Gini_Index'])

    def fn_8(self):

        flag = 0

        for i in  self.zscore:

            if i>3.0:

                print('Outlines Found. Zscore is -> ',i)

                flag = flag+1

        if flag==0:

            print('None Outlines Found With value more than 3.')

    def fn_9(self):

        self.corr_influence = self.sm.get_influence()

        (self.c, p) = self.corr_influence.cooks_distance

        plt.stem(np.arange( 20),np.round( self.c, 3 ),markerfmt=',')

    def fn_10(self):

        fig, ax = plt.subplots( figsize=(8,6))

        influence_plot( self.sm, ax = ax )

        plt.show()

count = country()
count.Question_1()
count.fn_2()

print('The R-Square Score is ->  ',count.r2_score)

print('The Root Mean Square Error is -> ',count.rms)

print('The Other Values are -> ',count.linreg)
count.Question_3()
count.fn_3()
count.fn_6()
count.display_graphs()

# Scatter Plot with Linear Regression and Showing 95% Confidence Interval
count.sm.summary2()
count.fn_4()

print('Normal Distribution of Residual is \n',count.corruption_resid.values)
count.fn_5()
count.fn_7()

print(count.zscore)

count.fn_8()

plt.stem(np.arange( 20),count.zscore,markerfmt=',')
count.fn_9()

print(count.c)
count.fn_10()
count.print_details()
count.display_corr()
count.display_graphs(orders = 5)