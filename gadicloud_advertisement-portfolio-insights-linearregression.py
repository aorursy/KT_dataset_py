!pip install bioinfokit
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv 0r pd.read_excel)
import datetime
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
import string
from keras.preprocessing.text import Tokenizer
import nltk
import pandas as pd
from keras.utils import np_utils
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Input, Dense, Flatten, Dropout, Embedding, LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras import regularizers, Sequential
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
import pickle
from keras.activations import softmax
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from mlxtend.plotting import plot_linear_regression
from scipy.stats import linregress
from bioinfokit.analys import stat
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

print('-------------------------------------------------------------------------------------')
print('------------------   Regression Analysis on Advertisement DataSet ------------------ ')
print('-------------------------------------------------------------------------------------')

print('-------------------------------------------------------------------------------------')
print(' As part of this dataset we will detail out the Z/T Statistics and p values addressing the impact of sales budget on the mediums available ')

df = pd.read_excel("/kaggle/input/regressads/Advertising.xlsx", sheet_name=0, index_col=0)
print(df)
# df is the dataframe 
#print(df.head(5))
# to find the no of rows and columns in the dataset3
#print(df.shape)

print(" ---------------- QUESTIONS TO ASK -------------------- ")
print(" Question 1 : Is there a relationship between advertising budget and sales ?")
print(" Question 2 : How strong is the relationship between advertising budget and sales ?")
print(" Question 3 : Which media contribute to sales ?")
print(" Question 4 : How accurate is the effect of each medium on sales ?")
print(" Question 5 : How accurately can we predict future sales ?")
print(" Question 6 : Is the relationship linear ? ")
print(" Question 7 : Is the Interaction(synergy) effect among the advertising media ?")
print(" ------------------------------------------------------ ")
print(" Will be using LInear regression to understand the Predictor and Response variable")
print(" Sales could be related as(Simple Linear Regression), Sales =  B0 + B1 * TV, Sales =  B0 + B1 * Newspaper, Sales = B0 + B1 * Radio")
print("Getting the B0 & B1 coefficients calculated for Slope and Intercept.")
print("Plotting the data points and calculating the Least Squares for data points.")
print("Calculating the residuals(e) for e(i) =  y(i) - y^(i)")
print("RSS =  Residual sum of Squared = e1pow2 + e2pow2 + e3pow2 + .... + enpow2")
print(" e1pow2 =  (y1 - (B^0 + B^1 * x1))pow2")
print(" e1pow2 =  (y1 - B^0 - B^1 * x1)pow2")
print(" thus RSS = (y1 - B^0 - B^1 * x1)pow2 + (y2 - B^0 - B^1 * x2)pow2 + .... + (yn - B^0 - B^1 * xn)pow2")
print("------------- Calculaitng B^0 and B^1 ---------------")
print("B1 =  Σ(i=1 to n)(xi - X)(yi - Y)/Σ(i=1 to n)(xi - X)pow2")
print("B0 =  Y - B^1 X")
print("Y = 1/nΣ(i=1 to n)y(i), X = i/nΣ(i = 1 to n)x(i) are sample means")
print("-----------------------------------------------------")
Y = np.array(df['Sales'].tolist())
X = np.array(df['TV'].tolist())
print(Y)
print(X)
#Calculating the intercept, slope, p value , standard error
linregress(X, Y)
print(" ------------------ CALCULATING THE COEFFICIENT AND THE P VALUE ------------------- ")
print("B0 =  Intercept(intercept on the Y axis) = 0.0475")
print("B1 =  Slope = 7.03")
print("r value = 0.78")
print("p value= 1.4673897001948012e-42 which is less then p < 0.0001, good for accepting the Null Hypothesis")
print("Standard Error =  STDDEV/ root(n) =  0.00269")
f, ax = plt.subplots(figsize=(15, 10))
sns.regplot(x="TV", y="Sales", data=df, ax=ax);
# plotting more then 1 below graph
#sns.pairplot(tips, x_vars=["total_bill", "size"], y_vars=["tip"],hue="smoker", height=5, aspect=.8, kind="reg");
print("Understanding the Coefficients estimated from the regression for business value when B0 =  Slope = 0.0475, B1  = 7.03")
print("With the approximation generated from the advertisement on TV on sales budget, we can see that ,") 
print("If one spends $1000 more on TV advertisement then it can sell 0.0475 * 1000 = 47.5 more no of units of the product.")
print("Lets Access the Accuracy of coefficients B0 and B1")
print("Y = B0 + B1 * X + e, where B1(Slope) = Average increase in Y with one unit increase in X, e =  random error")
print("Since we are considering the Sample from the sales data(not the Population data), we are uncertain about how the sample mean(M^) will estimate the Mean(M).")
print("The estimate of M can be biased, be it underestimated or overestimated.")
print("The sample mean derives here that, the M^(mu hat) from one sample data can onver estimate the M(mu), and the M^ from one sample data can under estimate M")
print("Next step is calculating how accurate is the M^ to M, we use STANDARD ERROR to answer this.")
print("Standard Error defines as SE(M^) =  SD/Squareroot(n), SD = Standard Deviation, n = no of records")
print("To understand the estimated Coefficients(B0^ & B1^) and the True Coefficients(B0 and B1), we tend to calculate the Standard Error between the coefficients")
print("SE(B0^) = SD * [1/n + Xpow(2)/Σi=1 to n (x-X)pow2]")
print("SE(B1^) = SD / ΣX/i=1 to n(x-X)pow2")
print("One can see that the SE(M) (M = sample mean) will be equal to SE(B0^) when X is ZERO, This estimate s called as RSE or Residual Standard Error")
print("RSE =  SqRt(RSS / (n-2))")
print("Note : - we are considering the impact of sales with respect to advertisement on TV")
print("The Standard Errors can help in calculating the Confidence Intervals, which defines the lower and upper range of values with any specified Confidence Level")
print("Calculating what values of B0^ and B1^ lie in their range with a confidence of 95%")
print("Note: not considering the Z/T statistics here as we are calculatng the SE for coefficients")
print("Thus -  CI(95%) states for B1^ as B1^ +- 2*SE(B1^), can be represented as [B1^ - 2 * SE(B1^), B1^ + 2 * SE(B1^)]")
print("Thus -  CI(95%) states for B0^ as B0^ +- 2*SE(B0^), can be represented as [B0^ - 2 * SE(B0^), B0^ + 2 * SE(B0^)]")
print("Calculate the CI(95%) for the range of B0^(Intercept) and B1^(Slope), we get B0(95%)  = [6.130,7.935], B1(95%) = [0.042,0.053]")
print("-----------------------    One can conclude for(TV advertisement)   ----------------------- ")
print(" -----    (B0) In the absence of any adevertisement, the sales will fall by 6130 - 7935 units")
print(" -----    (B1) for increase in 1000 $ on TV advertisement, one can increase the sales by 42 - 53 units of the product ")
print("Utilising the Standard Error for Hypothesis Analysis")
print("For the varibles Response(Y) and Predictor(X), are being related:")
print(" If relationship Doesnt exists then Null Hypothesis results in H(0) : B1 = 0, Note : accepting the Null Hypothesis for B1 = 0")
print(" If relationship exists then alternate Hypothesis results in H(a) : B1 /= 0, Note : the alternate Hypothesis for B1 /= 0")
#using bioinformatics for stats

df_Statistics =  df
data_RegresionDetails = stat()
data_RegresionDetails.lin_reg(df=df_Statistics, x=["TV"], y=["Sales"])
print(" -------------------- USING THE BIOINFORMATICS STATS FOR REGRESSION ANALYSIS ------------------- ")
print("Note 1 - An increase in 1000 $ in TV advertisement, wil increase the sales by 47.5 units of product")
print("Note 2 - We can see that the Estimate(Coefficient) for B0 and B1 are very large relative to their Standard Error, thus the T Statistics are also large")
print("Note 3 - We could see that the P Value for both coefficients are Non Zero, which REJECTS the Null hypothesis")
print("Cont.. Thus Rejecting the NULL Hypothesis says, there exists relationship between Advertisement Medium and Sales")
print(" ------------- Accessing the quality of Linear Regression Fit ----------------------- using RSE which measures the 'LACK OF MODEL FIT'")
print(" Accessing quality of linear regression using 1) RSE(Residual Standard Error)  2) Rpow(2)")
print(" RSE is an estimate of Standard Deviation of e")
print(" RSE =  sqrt(RSS/(n-2)), RSS = epow2 =  Σi=1 to n(yi - y^i)pow2")
print(" From the Results we see -  RSE = 3.2587")
print(" we know that the total mean of sales(Y) = 14022, what do we conclude when RSE =  3258")
print(" Even if the coefficients B0 and B1 were known to be exact, any prediction on sales based TV adevertisement would be OFF by 3258 units on an average")
print(" Thus the percentage Error of movement will be = 3258/14022 = 23%, THus for te model to fit well, the RSE has to be SMALL")
print("During Calculation of RSE, we saw that is was dependent on Y as it uses its mean, the R2 on the other hand is independent of Y")
print(" Adjusted r-squared(R2) = 0.6099")
print(" Calculating R2 =  (TSS - RSS)/TSS")
print(" R2 closed to indicates a large proportion of variablity in the response. R2 near to 0 indicates that the regression did not explain much of the variability in the response, because the linear model is wrong.")
print(" Till here it was about using Simple Linear Regression considering the mediums individually to check the regress of Y on X  ")
print("                                ")
print(" -------- Next to Cover Multiple Regression - where the mediums are taken together to Regress Y on X1, X2 & X3 ------- ")
# plotting more then 1 below graph
sns.pairplot(df, x_vars=["TV", "Radio", "Newspaper"], y_vars=["Sales"],height=5, aspect=.8, kind="reg");
print(" -- We define separate slope coefficients for each predictor --")
print(" The Equation can be termed as  : Y  =  B0 + B1 * X1 + B2 * X2 + ..... + Bn * Xn + e(error)")
print(" In the use case the Y Regress on X becomes, Sales =  B0 + B1*TV + B2*Radio + B3*Newspaper + e")
print(" We know the Coefficients B0 , B1, B2, B3 must be estimated for the regression model to fit.")
print(" Assumpation : let the estimates to B^0, B^1, B^2, B^3 ")
print(" The parameters need to be estimated using LEAST SQUARE APPROACH")
print(" lets minimise the RSS(Residual Sum of Squared) to estimate the coefficients")
print(" -----------------------------------------------")
print(" we know RSS  = Σi=1 to n (epow2) ")
print(" e = (y - y^), RSS =  Σi= 1 to n(yi - y^i)pow2")
print(" RSS =  Σi= 1 to n (yi -  B^0 - B^1 * xi1 -  B^2 * xi2 - ..... - B^p * xip)pow2")
#using bioinformatics for stats
# All 3 Predictors considered
df_MultiRegressStatistics =  df
data_MultiRegresionDetails = stat()
data_MultiRegresionDetails.lin_reg(df=df_MultiRegressStatistics, x=["TV","Radio","Newspaper"], y=["Sales"])
print(" -------------------- USING THE BIOINFORMATICS STATS FOR MULTI REGRESSION ANALYSIS ------------------- ")
print(" The Estimated Coefficients are B0(Intercept):2.94, B1(Coefficient):0.0458, B2(Coefficient):0.1885, B3(Coefficient):-0.001")
print(" Interpreting the results for Regression equation: 2.9389 + (0.0458*TV) + (0.1885*Radio) + (-0.001*Newspaper)")
print(" Considering the maximum units of the products to be sold, we consider coefficient with bigger value which is B2 here")
print(" For every $1000 spent on TV advertisement, the system can expect average in sales of product by 189 units.")
print(" ----Comparing the Estimated Coefficients from Simple Linear Regression to Multiple Linear Regression coefficients ----")
print(" ----------------------------------------------------------------------------------------------------------------------")
print(" Coefficients from Simple Linear Regression - B1(TV): 0.0475, B2(Radio): 0.203, B3(Newspaper): 0.055")
print(" Coefficients from Multiple Linear Regression - B1(TV): 0.0458, B2(Radio): 0.1885, B3(Newspaper): -0.001")
print(" ----------------------------------------------------------------------------------------------------------------------")
print("Note 1: ")
print(" We can notice that the B1 and B2 values from both the regression are very much similar")
print(" Whereas the coefficient(B3- Newspaper) is negative and the corresponding the P Value for Coefficient B3 is also no longer significant")
print("Note 2: ")
print(" Understanding the correlation matrix data between the Predictors for further analysis.")
print(" As one predictor can be an enabler for another predictor to do well in sales")
df_MultiRegressStatistics.corr() 
corr_HeatShow = df_MultiRegressStatistics.corr()
corr_HeatShow.style.background_gradient(cmap='coolwarm')
print(" One can see from the correlation matrix sheet above")
print(" The Correlation between Radio & Newspaper, shows to be the strongest which is 0.35")
print(" The observation says in market where one spends on Radio, in those markets one can also boost sales by spending on newspaper advertisement")
print(" The sales from Newspaper advertisement act as surrogate to Radio advertisement")
print("Note 3: Checking for relationship in Simple Linear Regression")
print("There is Relationship between X and Y when B1 NOT EQUAL to 0")
print("Note 4: In case of Multiple Linear Regression there is relationship between predictors when  atleast of coefficients B1, B2, B3,..., Bn NOT EQUAL to 0")

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Checking for relationship between the Predictor and the Response using F - Statistics")
print("SCENARIO 1: Another way to test the relationship between Response and Predictor")
print(" Considering a case where one has large no predictors say =  100(X1, X2....Xn), n = 200 ")
print("In this case in order to accept the NULL Hypothesis we have H0 : B0 = B1 =..= Bp = 0")
print("Which states all the coefficients estimated are 0 and certainly there isnt any relationship between Predictors and Response")
print("In this case atleast 5% of the p-values associated with the Predictors will be below 0.05 also can be termed as almost guaranteed that we will have 5 small p-values in the absence of RELATIONSHIP")
print("Thus with HIGH no of predictors, there is every chance that we cannot say there is a relationship between Predictors and Response")
print("Whereas the F Statistic does not suffer from this set of issues, when we have HIGH no of predictors, trying to bind relationship with Response")
print("Thus with H0 : B0 : B1 : B2 = 0 , which says we ACCEPT the Null Hypothesis, the with F Statistics , there is only 5% chance that F Statistics will result in p-value below 0.05, be any no predictors")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("SCENARIO 2: when the Predicors(P) are LARGE say 300 and the Number of Samples(N) be 200, here P > N  ")
print("Which states we have more coefficients to be estimated, then the number of samples")
print("In such case we cannot fit the Multiple Linear Regression model using LEAST Squares, and thus F-Statistics cannot be used")
print("For such scenarios we need to use the FORWARD Selection, BACKWARD Selection, Mixed Selection concepts")
print("Consider Regress of Y on X1,X2....Xn, and then removing coefficients with high p-value and chekcing mix and match of all the estimates of Predictors on Y")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


#using bioinformatics for stats
# Considering only 2 Predictors to check on RSE and R2
df_MultiRegressStatistics_2Predictors =  df
data_MultiRegressStatistics_2Predictors = stat()
data_MultiRegressStatistics_2Predictors.lin_reg(df=df_MultiRegressStatistics_2Predictors, x=["TV","Radio"], y=["Sales"])
print(" -------------------- USING THE BIOINFORMATICS STATS FOR MULTI REGRESSION ANALYSIS WITH 2 Predictors(TV , Radio) ------------------- ")
corr_HeatShow_2Pred = df_MultiRegressStatistics_2Predictors.corr()
corr_HeatShow_2Pred
corr_HeatShow_2Pred.style.background_gradient(cmap='coolwarm')
print(" ---------  Examining Correlation  ------------- ")
print(" - In the previous sections we were estimating the Coefficients to test the Null Hypothesis")
print(" Now we would be considering the MODEL FIT scenarios by considering the predictors and the respective R2 and RSE values")
print("--------------------------------------------------------")
print(" ~~~~~~~~~  While considering 2 Predictors(TV, Radio) ~~~~~~~~~  ")
print("We got R2(Coefficient of determination (r-squared)) : 0.8972 ")
print("We got RSE(Residual standard error)(e =  (yi-y^i)pow2) : 1.6814")
print("--------------------------------------------------------")
print(" ~~~~~~~~~  While considering 3 Predictors(TV, Radio, Newspaper) ~~~~~~~~~  ")
print("We got R2(Coefficient of determination (r-squared)) : 0.8972 ")
print("We got RSE(Residual standard error)(e =  (yi-y^i)pow2) : 1.6855")
print("--------------------------------------------------------")
print("-- MODEL FIT : SCENARIO : Considering R2")
print("R2 here is the Sqaure of Cor(y,y^)pow2")
print("We can see that while considering the R2 for 2 predictors(TV , Radio) it was 0.8972")
print("We can see that while considering the R2 for all predictors(TV , Radio, Newspaper) it was 0.8972")
print("We can see that while considering the R2 for 1 predictors(TV) it was 0.6119")
print("we can conclude that, with addition of Radio to TV advertisement the R2 (correlaiton) has increased, but with addition of Newspaper to the advertisement, the R2 remains same or one can see a minor change in R2")
print("-- MODEL FIT : SCENARIO : Considering RSE(Residual Standard Error) , THE RSE has to decrease by adding the Predictors to the line")
print(" RSE = sqrt(RSS/n-1)")
print("We can see that while considering the RSE for 1 predictors(TV) it was 3.2587")
print("We can see that while considering the RSE for 2 predictors(TV , Radio) it was 1.6814")
print("We can see that while considering the RSE for all predictors(TV , Radio, Newspaper) it was 1.6855")
print("We can see that when only TV predictor was being used the RSE was 3.25, then with 2 Predictors(TV, Radio) the RSE became 1.6814")
print("With addition of 3rd Predictor(TV, Radio, Newspaper) the RSE became 1.6855")