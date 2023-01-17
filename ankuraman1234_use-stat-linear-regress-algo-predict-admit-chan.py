import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
from datetime import date
import pandas as pd 
import sklearn 
import warnings
from sklearn import linear_model
from sklearn.model_selection import train_test_split
warnings.simplefilter(action = "ignore", category  = 'Futurewarning')
%matplotlib inline
mydata = pd.read_csv("../input/Admission_Predict.csv")
mydata.head()
mydata2 = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
mydata2.head()
mydata2 = mydata2.iloc[400:500]
mydata2.head()
mydata = mydata.append(pd.DataFrame(data = mydata2))
mydata.info()
sns.pairplot(mydata)
# Drop Sl_No from Table
mydata = mydata.drop('Serial No.',axis = 1)
mydata.head()
mydata.rename(columns={'GRE Score': 'GRE_Score'}, inplace=True)
mydata.rename(columns={'TOEFL Score': 'TOFEL_Score'}, inplace=True)
mydata.rename(columns={'University Rating': 'University_Rating'}, inplace=True)
mydata.rename(columns={'Chance of Admit ': 'Chance_of_Admit'}, inplace=True)
mydata.info()
# Change Datatype Of Columns:- 
mydata.University_Rating = mydata.University_Rating.astype('object')
mydata.Research = mydata.Research.astype('object')
# divide data in Numeric and Cat variable
cat_var = [key for key in dict(mydata.dtypes)
             if dict(mydata.dtypes)[key] in ['object'] ] # Categorical Varible

numeric_var = [key for key in dict(mydata.dtypes)
                   if dict(mydata.dtypes)[key]
                       in ['float64','float32','int32','int64']] # Numeric Variable

# check any Extreme value is present in numeric variable
mydata.describe()
mydata.boxplot(column= numeric_var)
sns.catplot(x="University_Rating", y="Chance_of_Admit",
            kind="box", dodge=False, data= mydata);
sns.catplot(x="Research", y="Chance_of_Admit",
            kind="box", dodge=False, data= mydata)
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
# Select imprortant varaible in Catagorical
#ANOVA F Test COVERAGE
model = smf.ols(formula='Chance_of_Admit ~ University_Rating', data=mydata)
results = model.fit()
print (results.summary())
# Here The F-statistic is  high
# p-value is to low.
model = smf.ols(formula='Chance_of_Admit ~ Research', data=mydata)
results = model.fit()
print (results.summary())
# Here The F-statistic is not so high
# p-value is to low.
# Start Some Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
mydata.head(n = 3)
X = mydata.iloc[:,[0,1,3,4,5]]  #independent columns
y = mydata.iloc[:,-1]    #target column i.e price range
#apply SelectKBest class to extract top 10 best features
test = SelectKBest(score_func=chi2, k=2)
y = y.astype('int')
fit = test.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
features = fit.transform(X)
print(features[0:4,:])
# We Create dummy variable of both catagrical variable
D3 = pd.get_dummies(mydata.University_Rating, prefix='University_Rating').iloc[:,1:5]
D3.head(n = 5)
D4 = pd.get_dummies(mydata.Research, prefix='Research').iloc[:,1:2]
D4.head(n = 5)
mydata.head(n = 2)
# Now Add the All data make Final Data:- 
N1 = mydata[['GRE_Score','TOFEL_Score','SOP','LOR ','Chance_of_Admit']]
# Do Cbind in Data
New_Final_Data =  pd.concat([D3,D4,N1], axis=1)
New_Final_Data.reset_index(drop= True).head()
# 1. Linear Reggression
X = New_Final_Data.iloc[:,[0,1,2,3,4,5,6,7,8]]  #independent columns
y = New_Final_Data.iloc[:,-1] #dependent columns.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predict_train = lm.predict(X_train)
predict_test  = lm.predict(X_test)
# R- Square on Train
lm.score(X_train, y_train)
# R- Square on Test
lm.score(X_test,y_test)
# Mean Square Error
mse = np.mean((predict_train - y_train)**2)
mse
mse = np.mean((predict_test - y_test)**2)
mse
# MAPE in Train:- 
np.mean((predict_train - y_train)/y_train)
# MAPE in Test:- 
np.mean((predict_test - y_test)/y_test)
sns.regplot(x=y_train,y=predict_train,fit_reg=False)
sns.regplot(x = y_test, y = predict_test, fit_reg = False)

