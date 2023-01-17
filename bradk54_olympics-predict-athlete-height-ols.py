#loadpackages & data print data head
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#load data
df = pd.read_csv('../input/athlete_events.csv')
df.head(3)
#plot continuous data , perhaps this can be predicted using other metrics...spoiler
_=sns.violinplot(x='Sex',y='Height',data=df)
plt.title('Heights of Male and Female Athletes')
plt.show()
#load sklearn data set for regression predictions
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import statsmodels.api as sm
from scipy import stats
np.random.seed(17)
#prep data
df_na = df.dropna(subset=['Height','Weight','Age','Team','Sex','Sport'])
df_na['RAND'] = np.random.randint(1,3,df_na.shape[0])
holdOut = df_na[df_na['RAND']==1]## holdout set
df_na = df_na[df_na['RAND']!=1]
Height = df_na['Height']
data = df_na[['Age','Weight','Sex','Sport']]
data = pd.get_dummies(data,columns=['Sex','Sport'])

#split data
XTrain,XTest,yTrain,yTest = train_test_split(data,Height,test_size=0.2,random_state=2)

#Train model
lm = LinearRegression()
lm.fit(XTrain,yTrain)
ypred=lm.predict(XTest)
print(data.shape)

#https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
#code i pulled off line to help finding if features are statistically significant
X=XTrain
y=yTrain
params = np.append(lm.intercept_,lm.coef_)
predictions = lm.predict(X)

#newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
#MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))

# Note if you don't want to use a DataFrame replace the two lines above with
newX = np.append(np.ones((len(X),1)), X, axis=1)
MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
sd_b = np.sqrt(var_b)
ts_b = params/ sd_b

p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]

sd_b = np.round(sd_b,3)
ts_b = np.round(ts_b,3)
p_values = np.round(p_values,3)
params = np.round(params,4)

myDF3 = pd.DataFrame()
myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilites"] = [params,sd_b,ts_b,p_values]
#myDF3['features']=data.columns
myDF3.to_csv('myDF3.csv')
data.head(2).to_csv('features.csv')
print(myDF3)
#we definitly have a significant model lets take a look at its performance
print("Training R2: ",np.round(lm.score(XTrain,yTrain),4))
print("Testing R2: ",np.round(lm.score(XTest,yTest),4))

#predict on hold out set
holdOutHeight = holdOut['Height']
holdOutData = holdOut[['Age','Weight','Sex','Sport']]
holdOutData = pd.get_dummies(holdOutData,columns=['Sex','Sport'])
lm.predict(holdOutData)
print("Holdout: ",np.round(lm.score(holdOutData,holdOut['Height']),4))
print("Data set size: ", holdOutData.shape)
residuals = holdOutHeight - lm.predict(holdOutData)
_=sns.scatterplot(x=holdOut['Height'],y=residuals,palette='muted')
sns.set()
plt.show()

