# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import linear_model as lm

from sklearn.metrics import mean_squared_error, r2_score

from pandas.tools import plotting

from scipy import stats as stats

from scipy.stats import kurtosis, skew

from scipy.stats import chisquare, chi2_contingency, chi2

from scipy.stats import pearsonr

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

plt.style.use("ggplot")

import warnings

warnings.filterwarnings("ignore")





import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/TermPaper1.csv",header=0,sep=';')

data.head()
data.columns

data.describe()
x1=data.WDI

x2=data.TDI

x3=data["CONC."]

x4=data.RISK



print( 'excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(x1) ))

print( 'skewness of normal distribution (should be 0): {}'.format( skew(x1) ))



print( 'excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(x2) ))

print( 'skewness of normal distribution (should be 0): {}'.format( skew(x2) ))



print( 'excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(x3) ))

print( 'skewness of normal distribution (should be 0): {}'.format( skew(x3) ))



print( 'excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(x4) ))

print( 'skewness of normal distribution (should be 0): {}'.format( skew(x4) ))

plt.figure(figsize = (10,10))

df = pd.DataFrame(data,

                  columns=['BW','INCOME', 'EDUCATION', 'WDI', 'TDI','CONC.','RISK'])

boxplot = df.boxplot(column=['BW','INCOME', 'EDUCATION', 'WDI', 'TDI','CONC.','RISK'])
plt.figure(figsize = (5,5))

df = pd.DataFrame(data,

                  columns=['CONC.'])

boxplot = df.boxplot(column=['CONC.'])
plt.figure(figsize = (5,5))

df = pd.DataFrame(data,

                  columns=['RISK'])

boxplot = df.boxplot(column=['RISK'])


#This histogram has been made for the RISK outputs depended on the AREA types



#Since there are 34 columns it is appropriate to use about squaroot of the 34, nearly 6

u=plt.hist(data[data["AREA"]==1].RISK,bins=6,fc=(0,0,0.1,0.5),label = "Urban")

n_u=plt.hist(data[data["AREA"] == 2].RISK,bins=6,fc = (1,0,0,0.5),label = "Non-Urban")



plt.legend()

plt.xlabel("RISK")

plt.ylabel("Frequency")

plt.title("RISK Histogram for the Urban and Non-Urban Areas")

plt.show()



max_RISK_Urban = u[0].max()

max_RISK_Non_Urban = n_u[0].max()

index_RISK_Urban = list(u[0]).index(max_RISK_Urban)

index_RISK_Non_Urban = list(n_u[0]).index(max_RISK_Non_Urban)

most_RISK_Urban = u[1][index_RISK_Urban]

most_RISK_Non_Urban = n_u[1][index_RISK_Urban]

print("Most frequent RISK for Urban and Non-Urban areas are: ",most_RISK_Urban,"and",most_RISK_Non_Urban)

f=plt.hist(data[data["GENDER"]==1].RISK,bins=6,fc=(0,0,0.1,0.5),label = "Female")

m=plt.hist(data[data["GENDER"] == 2].RISK,bins=6,fc = (1,0,0,0.5),label = "Male")



plt.legend()

plt.xlabel("RISK")

plt.ylabel("Frequency")

plt.title("RISK Histogram for the Male and Female Gender Types")

plt.show()



max_RISK_Female = f[0].max()

max_RISK_Male = m[0].max()

indexFemale = list(f[0]).index(max_RISK_Female)

indexMale= list(m[0]).index(max_RISK_Male)

most_RISK_Female = f[1][indexFemale]

most_RISK_Male = m[1][indexMale]

print("Most frequent RISK for the Female and Male genders are: ",most_RISK_Female,"and",most_RISK_Male)
f=plt.hist(data[data["BW"]<=70].RISK,bins=6,fc=(0,0,0.1,0.5),label = "Under Average")

m=plt.hist(data[data["BW"] >= 70].RISK,bins=6,fc = (1,0,0,0.5),label = "Above Average")



plt.legend()

plt.xlabel("RISK")

plt.ylabel("Frequency")

plt.title("RISK Histogram Depending on the Average Weight")

plt.show()



max_RISK_UA = f[0].max()

max_RISK_AA = m[0].max()

indexUA = list(f[0]).index(max_RISK_UA)

indexAA= list(m[0]).index(max_RISK_AA)

most_RISK_UA = f[1][indexUA]

most_RISK_AA = m[1][indexAA]

print("Most frequent RISK for the body weights of people depending on the under or above the 70kg are: ",most_RISK_UA,"and",most_RISK_AA)
f=plt.hist(data[data["EDUCATION"]<=4].RISK,bins=6,fc=(0,0,0.1,0.5),label = "Under High School")

m=plt.hist(data[data["EDUCATION"] >4].RISK,bins=6,fc = (1,0,0,0.5),label = "Above High School")



plt.legend()

plt.xlabel("RISK")

plt.ylabel("Frequency")

plt.title("RISK Histogram Depending on EDUCATION")

plt.show()



max_RISK_UA = f[0].max()

max_RISK_AA = m[0].max()

indexUA = list(f[0]).index(max_RISK_UA)

indexAA= list(m[0]).index(max_RISK_AA)

most_RISK_UA = f[1][indexUA]

most_RISK_AA = m[1][indexAA]

print("Most frequent RISK for the High School or less and college or more  are: ",most_RISK_UA,"and",most_RISK_AA)
f=plt.hist(data[data["INCOME"]<=4].RISK,bins=6,fc=(0,0,0.1,0.5),label = "Normal-Poor")

m=plt.hist(data[data["INCOME"] >4].RISK,bins=6,fc = (1,0,0,0.5),label = "Normal-Wealthy")



plt.legend()

plt.xlabel("RISK")

plt.ylabel("Frequency")

plt.title("RISK Histogram Depending on Income")

plt.show()



max_RISK_UA = f[0].max()

max_RISK_AA = m[0].max()

indexUA = list(f[0]).index(max_RISK_UA)

indexAA= list(m[0]).index(max_RISK_AA)

most_RISK_UA = f[1][indexUA]

most_RISK_AA = m[1][indexAA]

print("Most frequent RISK depending on the income are: ",most_RISK_UA,"and",most_RISK_AA,"for the normal-poor and normal-wealthy people,respectively")
data_Urban = data[data["AREA"] == 1]

data_NonUrban = data[data["AREA"] == 2]

desc = data_Urban.RISK.describe()

Q1 = desc[4]

Q3 = desc[6]

IQR = Q3-Q1

lower_bound = Q1 - 1.5*IQR

upper_bound = Q3 + 1.5*IQR

print("Anything outside this range is an outlier: (", lower_bound ,",", upper_bound,")")

data_Urban[data_Urban.RISK < lower_bound].RISK

print("Outliers: ",data_Urban[(data_Urban.RISK < lower_bound) | (data_Urban.RISK > upper_bound)].RISK.values)
melted_data = pd.melt(data,id_vars = "AREA",value_vars = ['RISK', 'CONC.','WDI','EDUCATION','TDI','INCOME','AGE'])

plt.figure(figsize = (10,10))

sns.boxplot(x = "variable", y = "value", hue="AREA",data= melted_data)

plt.show()
# RISK factor for the Urban and Non-Urban Areas are zoomed in 



melted_data = pd.melt(data,id_vars = "AREA",value_vars = ['RISK'])

plt.figure(figsize = (10,10))

sns.boxplot(x = "variable", y = "value", hue="AREA",data= melted_data)

plt.show()
# RISK factor for the Urban and Non-Urban Areas are zoomed in 



melted_data = pd.melt(data,id_vars = "AREA",value_vars = ['CONC.'])

plt.figure(figsize = (10,10))

sns.boxplot(x = "variable", y = "value", hue="AREA",data= melted_data)

plt.show()
#Probability plot for the WDI parameter

stats.probplot(data.WDI, sparams=(), dist='norm', fit=True, plot=plt, rvalue=False)
#Probability plot for the Risk parameter

stats.probplot(data.RISK, sparams=(), dist='norm', fit=True, plot=plt, rvalue=False)
#Probability plot for the CONCENTRATION parameter

stats.probplot(data['CONC.'], sparams=(), dist='norm', fit=True, plot=plt, rvalue=False)
#Probability plot for the INCOME

stats.probplot(data.INCOME, sparams=(), dist='norm', fit=True, plot=plt, rvalue=False)
#Probability plot for the EDUCATION

stats.probplot(data.EDUCATION, sparams=(), dist='norm', fit=True, plot=plt, rvalue=False)
#Probability plot for the TDI parameter

stats.probplot(data.TDI, sparams=(), dist='norm', fit=True, plot=plt, rvalue=False)
a=stats.normaltest(data.TDI, axis=0, nan_policy='omit')

b=stats.shapiro(data.TDI)

c=stats.anderson(data.TDI, dist='norm')

print(a,"and Shapiro Results are :",b,"where the variables are test statistics and p-value, respectively")

print("Anderson Test Result is:",c)

print(

)

a2=stats.normaltest(data.WDI, axis=0, nan_policy='omit')

b2=stats.shapiro(data.WDI)

c2=stats.anderson(data.WDI, dist='norm')

print(a2,"and Shapiro Results are :",b2,"where the variables are test statistics and p-value, respectively")

print("Anderson Test Result is:",c2)

print(

)



a3=stats.normaltest(data.RISK, axis=0, nan_policy='omit')

b3=stats.shapiro(data.RISK)

c3=stats.anderson(data.RISK, dist='norm')

print(a3,"and Shapiro Results are :",b3,"where the variables are test statistics and p-value, respectively")

print("Anderson Test Result is:",c3)

print(

)





a4=stats.normaltest(data.EDUCATION, axis=0, nan_policy='omit')

b4=stats.shapiro(data.EDUCATION)

c4=stats.anderson(data.EDUCATION, dist='norm')

print(a4,"and Shapiro Results are :",b4,"where the variables are test statistics and p-value, respectively")

print("Anderson Test Result is:",c4)

print(

)



a5=stats.normaltest(data['CONC.'], axis=0, nan_policy='omit')

b5=stats.shapiro(data['CONC.'])

c5=stats.anderson(data['CONC.'], dist='norm')

print(a5,"and Shapiro Results are :",b5,"where the variables are test statistics and p-value, respectively")

print("Anderson Test Result is:",c5)

print(

)



a6=stats.normaltest(data.INCOME, axis=0, nan_policy='omit')

b6=stats.shapiro(data.INCOME)

c6=stats.anderson(data.INCOME, dist='norm')

print(a6,"and Shapiro Results are :",b6,"where the variables are test statistics and p-value, respectively")

print("Anderson Test Result is:",c6)

print(

)
a=data.groupby('GENDER').size()

print(a)

a.plot.pie(figsize=(4,4))

a=data.groupby('AREA').size()

print(a)

a.plot.pie(figsize=(4,4))
a=data.groupby('EDUCATION').size()

print(a)

a.plot.pie(figsize=(4,4))
a=data.groupby('INCOME').size()

print(a)

a.plot.pie(figsize=(4,4))
# Categorizing by gender, some descriptive parameters have been given here.  



f=data[data["GENDER"]==1]

m=data[data["GENDER"]==2]

f.head()

female_Risk_Mean=f.RISK.mean()

F_Median=np.median(f.RISK)

male_Risk_Mean=m.RISK.mean()

M_Median=np.median(m.RISK)

#Print Median 

print("Median of the Risk for the female and male types are:",F_Median,"and",M_Median)

#Standard Deviaton of F and M

F_std=np.std(f.RISK)

M_std=np.std(m.RISK)

print("Standard deviation of the Risk for the female and male types are:",F_std,"and",M_std)

#Print Mean Values

print("The average risk for the female and male types are:",female_Risk_Mean,"and",male_Risk_Mean,"And the given critical Level is", 0.001)

#Population Risk

Pop_risk=data.RISK.mean()

Pop_risk_std=np.std(data.RISK)

Pop_risk_median=np.median(data.RISK)

print("mean-std-median values of the population are:",Pop_risk,Pop_risk_std,"and",Pop_risk_median)

df=data.groupby('GENDER',axis=0).mean()

df
df=data.groupby('EDUCATION',axis=0).mean()

df

df=data.groupby('AREA',axis=0).mean()

df
df=data.groupby('INCOME',axis=0).mean()

df
#H0:mü=1x10^-4

#H1:mü>1x10^-4, One sided alternative hypothesis

#alfa=0.05

mü=0.0001

s=Pop_risk_std

print("Standard deviation of the sample is",s)

t0=(Pop_risk-mü)/(s/np.sqrt(35))

#alfa=P(T<t0)

print("t0 values is:",t0)

print("Q1-[g]","Conclusion: Since",t0,">1.691, null hyphothesis is rejected at the 0.05 level of significance that the average cancer risk level exceeds the acceptable carcinogenic level and p-value<0.005 from table IV,Appendix A, page 656")





#H0:ki2_0=ki2_alfa_v

#H1:ki2_0>ki2_alfa_v



sigma2=0.001

s=Pop_risk_std

n=35

ki2_0=(n-1)*s*s/(sigma2*sigma2)

ki2_alfa_v=49



print("Q1-[h]-1st Option","Conclusion: X^2_0=",ki2_0,">","X^2_(0.05,34)=",ki2_alfa_v,"There is strong evidence that null hypothesis is fail to rejected")
t=stats.chisquare(data.TDI)

s=stats.chisquare(data.RISK)

print("results for TDI and RISK are:",t,"and",s)
stat,p,dof,expected=chi2_contingency(data.RISK)

stat,p,dof,expected
CI=0.95

critical=0.001

if abs(stat)>=critical:

    print('Dependent(reject H0)')

else:

        print('Independent(fail to reject H0)',"and p-Value is:",p)

print("Q1-[h]-2nd Option: Null hypothesis is fail to rejected with strong evidence")
#Correlation between parameters

f,ax=plt.subplots(figsize = (18,18))

sns.heatmap(data.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.title('Correlation Map')

plt.savefig('graph.png')

plt.show()
plt.figure(figsize = (9,9))

plt.scatter(data.WDI,data.TDI,color='black')

plt.show()
regression_model =lm.LinearRegression()





regression_model.fit(X=pd.DataFrame(data.WDI),

                     y=data.TDI)

print("Model is: y=m+[n]x, where m is the intercept and n is slope")

print("Intercept of the plot is:",regression_model.intercept_)

print("Slope or regression model coefficient is:",regression_model.coef_)





regression_model.score(X = pd.DataFrame(data.WDI), 

                       y = data.TDI)

train_prediction = regression_model.predict(X = pd.DataFrame(data.WDI))



# Actual - prediction = residuals

residuals = data.TDI - train_prediction



residuals.describe()


SSResiduals = (residuals**2).sum()



SSTotal = ((data.TDI - data.TDI.mean())**2).sum()



# R-squared

R_square=1 - (SSResiduals/SSTotal)

print("SSR and SST are:",SSResiduals,"and",SSTotal,",respectively. So R-Square is:",R_square)
plt.scatter(data.WDI,data.TDI,color='black')



# Plot regression line

plt.plot(data["WDI"],      # Explanitory variable

         train_prediction,  # Predicted values

         color="blue")
plt.figure(figsize=(9,9))



stats.probplot(residuals, dist="norm", plot=plt)


def rmse(predicted, targets):

      return (np.sqrt(np.mean((targets-predicted)**2)))



RMSE=rmse(train_prediction, data["TDI"])

print('Root mean squared error is:', RMSE)
regression_model =lm.LinearRegression()





regression_model.fit(X=pd.DataFrame(data.BW),

                     y=data.RISK)

print("Model is: y=m+[n]x, where m is the intercept and n is slope")

print("Intercept of the plot is:",regression_model.intercept_)

print("Slope or regression model coefficient is:",regression_model.coef_)







regression_model.score(X = pd.DataFrame(data.BW), 

                       y = data.RISK)







train_prediction = regression_model.predict(X = pd.DataFrame(data.BW))



# Actual - prediction = residuals

residuals = data.RISK - train_prediction



residuals.describe()









SSResiduals = (residuals**2).sum()



SSTotal = ((data.RISK - data.RISK.mean())**2).sum()



# R-squared

R_square=1 - (SSResiduals/SSTotal)

print("SSR and SST are:",SSResiduals,"and",SSTotal,",respectively. So R-Square is:",R_square)





plt.scatter(data.BW,data.RISK,color='black')



# Plot regression line

plt.plot(data["BW"],      # Explanitory variable

         train_prediction,  # Predicted values

         color="blue")







plt.figure(figsize=(9,9))



stats.probplot(residuals, dist="norm", plot=plt)
def rmse(predicted, targets):

      return (np.sqrt(np.mean((targets-predicted)**2)))



RMSE=rmse(train_prediction, data["RISK"])

print('Root mean squared error is:', RMSE)