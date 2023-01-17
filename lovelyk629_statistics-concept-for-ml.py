# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


""""

***************************** TOPIC : Line of best fit ******************************************************





Line of best fit is line through data points that best express the relation among these data points.



For Line of best fit  sum of square of residual errors between data point and indivisual line is at minimum .

    

  In term of   mathematical equation :

        y=mx+c 

        m: slope

        c: intercept

       m=n* (sum (x*y)-sum(x)sum(y))/(n* sum (x*x)- (sum(x))^2)

       n: no of data point on scatter plot

       

    c:(sum(y)- m* sum(x))/n

"""
""""



 ******************************************** TOPIC  : RMSE *****************************************************



Root mean square error



RMSE tells us how data point is around line of best fit. 

in other word how residual are spread among best fit line . How actual values are diff. from predicted value .

RMSE is SD(standard deviation) of residuals .



Residuals : prediction error / or how much data point are away from regression line 

    

  RMSE =  sqrt((y(predict)-y(actual))^2/n)



*************************************************************************

To remember : error diff. ->then square -> take mean -> square root

************************************************************************



RMSE is always betwwn 0 and 1 .



*****  As much Lower the  RMSE value better the model prediction *******



"""
"""

********************************************* TOPIC : R square *************************************************



R square determine the  proportion of variance of  dependent variable that is  explained by the independent variable 

in a regression model

in other word , to what extent variance of one varibale explained by other variable .

if R^2  value is 0.50 that means 50% variance of dependent variable can be explained by model input(independent variable)



R^2 value range from 0 to 1 or  0% to 100% .



we can't totaly relie on r^2 for test model prediction is good or not , we need other statistical term as well .

Higher  the value of R^2  -> good the model prediction ..this statement is not always true ... 



Higher/Lower value of  R^2  does not tell anything about model prediction .



R^2 =1-(Explained variation)/(Total Variation)





********************** ******************** TOPIC : Adjusted R^2  **********************************************************



R^2 only works well in case of simple linear regression with one explainatory variable .

in case of regression made up of multiple independent variable R^2 must be adjusted .



Every  independent variable added to model always increase R^2 value  whether added variable actually enchancing the model or not .

so in that case model with more term may seem better fit just for fact that it has more term.



Here Adjusted R^2 play its role . 

it adjust the value and increase only when addition of new term enchance the model .



so in short , increasing the term / addition of variable  always increase R^2 value whereas in adjusted R^2 

it checks whethere model is being enchanced or not then accordingly increase the value otherwise not .



Formula :



adj R^2 = 1-[(1-R^2)(n-1)/(n-k-1)]



n= no. of data point

k= no. of independent variable



"""







"""



****************************************** TOPIC :- P-value  **********************************************************



P value :-



it represent the probability of occurance of given event .



Lower the p value  that mean we can reject null hypothesis .



*******************************************************



Null Hypothesis :- the event which are on trial is called null hypothesis .



Alternative Hypothesis :-it is the one which you will believe when null hypothesis is supposed to be false .



**************************************************************************

for ex :- 



pizza shop claim it deliver pizza in 30 mnt or less than 30 mnt .



but you have doubt on it you think its more than 30 mnt , then you conduct hypothesis test .



null hypothesis :- delivery time is max 30 mnt .(normal claim /or event on trial)



Alternative hypothesis :- deliver time is more than 30 mnt .



Now we calculate P-value .its found  P=0.001 .

Hence its very less, we can reject null hypothesis  .

Lower the P-value , mean Hypothesis is in favor of Alternative Hypothesis .

Hence pizz is deliver in more than 30 mnt .





*******************************************************



Type 1 error : Incorrect rejection of null hypothesis 

Type 2 error  : Incorect acceptance of null hypothesis



****************************************************



"""

"""

F1 Score



F1 score : It being use to measure test accuracy .

    its Harmonic mean of test precision and recall .

    

    F1= HM(recall ,Precision)

     =  2* (precision *Recall) /(Precision+Recall)

        

        

Precision : - precision is define as accuracy of judgement  , how precise  the value is .

              (true postive) /(true positive+ false positive)

        

Recall :-   recall is called sensitivity , is the ability to identify the number of samples that would really count 

            positive for specific attribute.

            

            = (true positive)/(true positive+ False negative)

            





"""

"""

***************************************** TOPIC :- ROC and AUC ***************************************



    ROC :- Receiver operating characteristics

           It is perfromance measure of classification model , which tell how model can distiguish between

           true positive and true negative .

           

            It  is created by plotting true positive rate against false positive rate .

             plotting  sensitivity  against 1-specificity

             plotting TPR again FPR

             

        ROC is curve and AUC(area under curve) is measure of seperability .

        HIgher the AUC better the model is predicting 0s as 0 and 1s as 1.

        Excellent model has AUC=1 which means good measure of seperability . prdeciting 0 as 0 and 1 as 1

        

        AUC=0 , which means worst measure of seperability . predicting 0 as 1 and 1 as 0

        

        """

        

        

            

    
""""

************************** confusion Matrix ******************



confusion matrix use to predict performance of classification model .

it is table of combination of 4 values ,



TP : true positive   predicted positive and it is true

FP : false positive  predicted positive and it is false (type 1 error) 

TN : true negative  predicted negative and it is true

FN : false negative predicted negative and it is false (type 2 error)

    

    confusion matrix used to calculate precision , recall , F1-score

    

    """