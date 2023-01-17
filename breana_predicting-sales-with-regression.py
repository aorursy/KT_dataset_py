# Import packages



import matplotlib.pyplot as plt 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Read the data in to a dataFrame

data=pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv", index_col=0)

data.head()



# Any results you write to the current directory are saved as output.
#Sales is the respone varaible



#lets look at the shape of the data. there are 200 observations or markets in the dataset

data.shape



#Overview of Linear Regression assumptions we will check:

    #1.) Your variables should be continuouse (interval or ratio type varibles)

    #2.) The assumption of a Linear Relationship between the independent and Dependent Varaible(s)

    #3.) No significant outliers, they can reduce the fit of the regression model 

    #4.) Independent Observations

    #5.) Your data needs to show homoscedasticity, the varaince of errors is the same across all levels of the IV

    #6.) The residuals(error) of the regression line should be normally distributed
#Check assumption 2

#Testing the assumptions of a Linear Relationship between the independent and dependent varaible (s)

    #If the relationship between the Independent Variable (IV) and Dependent Varaible (DV) is not linear, the results of regression will under-estimate the true relationship

    #This under-estimation can present 2 major problems

        #1.) an increased chance of a Type II error for that IV

        #2.) and with multiple regression an increases risk of Type I errors (over-estimation) for other IVs that share variance with that IV

    #How to test for linearity:

        #1.)Scatterplots

fig, axs=plt.subplots(1,3,sharey=True)

data.plot(kind='scatter', x='TV', y='Sales', ax=axs[0], figsize=(12,4))

data.plot(kind='scatter', x='Radio', y='Sales', ax=axs[1])

data.plot(kind='scatter', x='Newspaper', y='Sales', ax=axs[2])
#We'll start with a simple linear regression model to predict sales with TV ad spending



#Estimate the coefficients using the statsmodel package

import statsmodels.formula.api as smf



#Create a fitted model:

lm=smf.ols(formula='Sales~TV', data=data).fit()



#print the coefficients:

lm.params



#This can be interpretted by saying a unit increase in TV ad spending is associated with as 0.047537 until increase in sales
#We could manually calculate the predicted sales for a $50,000 TV advertising budget

7.032594+0.047537*50
#Now lets us the model to make the prediction for us:



#First we need to create a dataframe for the model to place the prediction in

frame=pd.DataFrame({'TV':[50]})

lm.predict(frame)
#Lets make predictions for our dataset

frame=pd.DataFrame(data.TV)



preds=lm.predict(frame)



#lets plot our observed data

data.plot(kind='scatter', x='TV', y='Sales')



#Lets add our least squares line for our predicted values

plt.plot(frame,preds,c='red', linewidth=2)
#Lets check out or confidence intervals

lm.conf_int()
#Lets check our p values to see if TV is significant:

lm.pvalues



#using a 95% confidence interval our pvalue is significant at the .05 level because it is well under .05
#So how well does out model fit the data??

#Lets do this by checking our R-squared value, this is the proportion of varaince explained by the model



lm.rsquared
#Now lets try some multiple regression

#We'll use the same data, but we'll add radio and newspaper to our model:



lm=smf.ols(formula='Sales~ TV + Radio + Newspaper', data=data).fit()



#And we'll print the coefficients

lm.params
#We'll take a look at the model summary too:

lm.summary()



#What can we get from this summary??

#1.) TV and Radio are significant at the .05 level and Newspaper is now

#2.) TV and Radio are positivly associated with Sales, and Newspaper is slightly negative

#3.) This model has a higher R-squared value (0.897) then the simple linaer regression model with just TV
#We'll remove Newsapaper from out model:

lm=smf.ols(formula='Sales ~ TV + Radio', data=data).fit()

lm.summary()
#lets add the predictions to our data set

data2=pd.DataFrame(data[['Radio','TV', 'Sales']])

data2['Predicted']=lm.predict(data2)

data2['Residuals']=data2['Sales']-data2['Predicted']



data2.plot(kind='scatter', x='Predicted', y='Residuals')

data2.plot(kind='scatter', x='TV', y='Residuals')