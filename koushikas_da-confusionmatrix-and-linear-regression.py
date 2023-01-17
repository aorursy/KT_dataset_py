import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
import statsmodels.api as sm1
df=pd.read_csv("../input/AirQualityUCI_req.csv",parse_dates=True)
df=df.drop(['Date','Time'],axis=1)
columns=(list(df))
print(columns)
#Removing Rows which have colum value =-200
print(df.shape)
df=df[df !=-200]
for col in columns:
    df=df[np.isfinite(df[col])]
print(df.shape)
a1 = sns.boxplot(data=df , orient="h", palette="Set2")
a1 = sns.boxplot(data=df[['CO(GT)','T','RH','AH','NO2(GT)']] , orient="h", palette="Set2")
a1 = sns.boxplot(data=df[['PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','PT08.S5(O3)']] , orient="h", palette="Set2")
a1 = sns.boxplot(data=df[['NMHC(GT)','NOx(GT)']] , orient="h", palette="Set2")
#z score helps us to describe SD and MEan of group of data points 
#in most case outliers lie outside 3,-3

z = np.abs(stats.zscore(df))
print(z)
print(z<3)

#Removing values with grater than 3 Z values (outliers)
print(df.shape)
df = df[(z < 3).all(axis=1)]
print(df.shape)
df.describe()
#Finding Correlation Matrix 
df.corr()
# # From the Corelation Matrix we can infer few things 
# - We find PT08.S5(O3) is strongly or moderately correlated with all gases. This observation
# can be attributed to the fact that ozone acts as catalyst to the reactions that result in the
# production of all these gases. This finding is also in line with that in [2].

# - CO(GT), PT08.S1(CO), C6H6(GT) and PT08.S2(NMHC) are not strongly correlated with
# nitrogen oxides measurements of both PT and the reference sensors but show strong
# correlation with parameters. This indicates existence of two groups one containing of all the
# hydrocarbon gases and the other containing Nitrogen oxides. This two groups are
# independent of each other.

# - The values of the new resistance sensors for NOx show fair correlation with the reference
# sensor values indicating that the new sensors are performing moderately well.

# - It is interesting to note PT08.S3(NOx) has negative yet a moderately strong correlation with
# NOx(GT) implying that the technology used in the new NOx sensors is different to exhibit
# consistently the same characteristic of NOx(GT) but in the negative form.  
y= df['NO2(GT)']
x= df.drop(['NO2(GT)'],axis=1)

modelAll = sm.OLS(y,x).fit()
modelAll.summary()
#from the sumary we can see that some variable are rrelated to NO2 
#by seeing that lot of variable have  low p value and F stat of overal model is 
# greater than 0 therefore null hypothesis can be elimnated 

#since T ,RH, AH, NMHC(GT), PT08.S3(NOx) has high p value and also correlation matrix also <0.8 we are going to 
#elimante from furtehr evaluation 
columns.remove('NO2(GT)')
columns.remove('T')
columns.remove('RH')
columns.remove('AH')
columns.remove('PT08.S3(NOx)')
columns.remove('NMHC(GT)')
print(columns)
def evaluateModel (model):
    print("RSS = ", ((y - model.predict())**2).sum())
    print("R2 = ", model.rsquared)
#forward Selection

for col in columns:
    x= df[col]
    model1 = sm.OLS(y,x).fit()
    print(col)
    evaluateModel(model1)
    print()
# Lowest Rss and highest R2 is PT08.S2(NMHC). Thereforere we choose This as our first input feature.    
X1='PT08.S2(NMHC)'
columns.remove('PT08.S2(NMHC)')
#forward Selection

for col in columns:
    x= df[[col,X1]]
    model1 = sm.OLS(y,x).fit()
    print(col)
    evaluateModel(model1)
    print()
# Lowest Rss and highest R2 is NOx(GT) Thereforere we choose This as our first input feature.    
X2='NOx(GT)'
# columns.remove('NOx(GT)')
x= df[[X1,X2]]
model = sm.OLS(y,x).fit()
model.summary()
# features = "+".join(df[[X1,X2]])
# print(features)
# y='NO2(GT)~'
# # get y and X dataframes based on this regression:
# Y1, X1 = dmatrices(y + features, df, return_type='dataframe')
X1=df[[X1,X2]]
Y1=df['NO2(GT)']
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]
vif["features"] = X1.columns
vif.round(1)
#since VIF factor> 10 indicating high collinearity between the two we have to use one of the 2 variables
#therefore we use one among the PTO8.S2(NMHC)
# therefore linear regression is 
x= df['PT08.S2(NMHC)']
model = sm.OLS(y,x).fit()
model.summary()

fig, ax = plt.subplots()
fig = sm1.graphics.plot_fit(model, 0, ax=ax)
ax.set_ylabel("NO2(GT)")
ax.set_xlabel("PT08.S2(NMHC)")
ax.set_title("Linear Regression")
plt.show()
# From the above graph and statistics, we can infer that:
# - Our linear regression model is properly fitted with R-squared value of 0.918.
# - F- statistic of 7419e+04 indicates that our model does not support null hypothesis.
# - The P value is almost 0 indicating that PT08.S5(O3) is another evidence against null
# hypothesis.

