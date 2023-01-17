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
#Market Mix Modeling

#Sales Data 

#To understand how much each marketing input contributes to sales, and how much to spend on each marketing input.



#import libraries

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns





df = pd.read_csv("../input/Advertising.csv")

df.head()
#remove extra 'Unnamed' column

df_clean = df.loc[:, ~df.columns.str.contains('^Unnamed')]

df_clean.head()
#Data Description 



df_clean.describe()
#Correlation b/w variables

corr = df_clean.corr()

sns.heatmap(corr)
#Labels and features

labels = df_clean['sales']

features = df_clean.drop(['sales'], axis=1)
# Scatter graph b/w response and features

for x in features:

    plt.plot(labels, features[x], 'ro')  # arguments are passed to np.histogram

    plt.title("Sales vs " + x)

    plt.xlabel(x)

    plt.ylabel("sales")

    plt.show()
#Data Distribultion 

for x in features:

    plt.hist(features[x], bins='auto')  # arguments are passed to np.histogram

    plt.title(x)

    plt.show()
#As from the above histogram graph, the data distribution for the newspaper is skrew towards left.

#Lets correct it using Box Cox which helps in removing the data skrewness.



from scipy import stats

import matplotlib.pyplot as plt



fig = plt.figure()

ax1 = fig.add_subplot(211)

x = df_clean['newspaper']

prob = stats.probplot(x, dist=stats.norm, plot=ax1)

ax1.set_xlabel('')

ax1.set_title('Probplot against normal distribution')

#We now use boxcox to transform the data so it’s closest to normal:

ax2 = fig.add_subplot(212)

df_clean['newspaper'], _ = stats.boxcox(x)

prob = stats.probplot(df_clean['newspaper'], dist=stats.norm, plot=ax2)

ax2.set_title('Probplot after Box-Cox transformation')



plt.show()
plt.hist(df_clean['newspaper'], bins='auto')  # arguments are passed to np.histogram

plt.title("Newspaper after Box cox transformation")

plt.show()
plt.plot(df_clean['sales'], df_clean['newspaper'], 'ro')  # arguments are passed to np.histogram

plt.title("Scatter plot b/w sales and newspaper")

plt.xlabel("Newspaper")

plt.ylabel("Sales")

plt.show()
# As from the above graph it is clear that newspaper do not have any relationship with the Sales.

# Lets build 2 algorithm with and without newspaper to get more clear picture.
import statsmodels.formula.api as sm

model1 = sm.ols(formula="sales~TV+radio+newspaper", data=df_clean).fit()

model2 = sm.ols(formula="sales~TV+radio", data=df_clean).fit()

model3 = sm.ols(formula="sales~TV", data=df_clean).fit()

#sales~TV+radio+newspaper

print(model1.summary())

print(model2.summary())

print(model3.summary())
#AIC BIC

#They are used to compare a number of models and the model with lowest values of AIC and BIC is 

#considered to be the best (however AIC and BIC are accompanied often by other tests of fit, 

#e.g. RMSEA, CFI, TLI etc. and the decision which model is the best is not based on AIC and BIC only



#Model 1

#model1 = sm.ols(formula="sales~TV+radio+newspaper", data=df_clean).fit()



#R-squared:                       0.897

#Adj. R-squared:                  0.896

#F-statistic:                     570.3

#Prob (F-statistic):           1.58e-96

#Log-Likelihood:                -386.18

#AIC:                             780.4

#BIC:                             793.6



#Model2

#model2 = sm.ols(formula="sales~TV+radio", data=df_clean).fit()

#R-squared:                       0.897

#Adj. R-squared:                  0.896

#F-statistic:                     859.6

#Prob (F-statistic):           4.83e-98

#Log-Likelihood:                -386.20

#AIC:                             778.4

#BIC:                             788.3

    

#Model3

#model3 = sm.ols(formula="sales~TV", data=df_clean).fit()

#R-squared:                       0.612

#Adj. R-squared:                  0.610

#F-statistic:                     312.1

#Prob (F-statistic):           1.47e-42

#Log-Likelihood:                -519.05

#AIC:                             1042.

#BIC:                             1049.



#From the above results it is clear that the 'model 2' with feature 'radio' and 'TV' is having the lowest

#AIC & BIC



#Model 2 Parameters, error, and r square

print('Parameters: ', model2.params)

print('R2: ', model2.rsquared)

print('Standard errors: ', model2.bse)
#Actual and predicted values

y_pred = model2.predict()

df1 = pd.DataFrame({'Actual': labels, 'Predicted': y_pred})  

df1.head(10)
#Final observation



#Values from Model 1 -> sales~TV+radio+newspaper



#==============================================================================

#                 coef    std err          t      P>|t|      [0.025      0.975]

#------------------------------------------------------------------------------

#Intercept      2.8894      0.361      7.995      0.000       2.177       3.602

#TV             0.0457      0.001     32.810      0.000       0.043       0.048

#radio          0.1876      0.008     22.190      0.000       0.171       0.204

#newspaper      0.0060      0.040      0.152      0.879      -0.072       0.084

#==============================================================================



#newspaper Values

#Coef   : 0.0060

#t-test : 0.152

#p-value: 0.860



#From the above values it is clear that newspaper maketing is not affecting sales by any chance.

#High Pvalue(>0.005) is always fail to reject null hypothesis.

#That means there is no relationship between the newspaper marketing and sales.



#=========================================Thank you ====================================================

#==================================Do comment your thoughts=============================================
