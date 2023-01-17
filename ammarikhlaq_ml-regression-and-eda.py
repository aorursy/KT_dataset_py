import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/insurance.csv')

df.dtypes
df.head(10)
df['sex'].value_counts()
df['smoker'].value_counts()
df['smoker']=df['smoker'].apply(lambda x: 0 if x=='no' else 1)
df['sex']=df['sex'].apply(lambda x: 0 if x=='female' else 1)
df.head()
p=np.arange(len(df['smoker'].unique()))
sum_of_smokers=len(df['smoker'])
non_smokers=0
smokers=0
for x in df['smoker']:
    if(x==0):
        smokers +=1
    elif(x == 1):
        non_smokers +=1
        
sm=[smokers,non_smokers]
percentage_of_smokers= "{0:.2f}".format((smokers/float(sum_of_smokers))*100)
percentage_of_non_smokers= "{0:.2f}".format((non_smokers/float(sum_of_smokers))*100)
plt.bar(p,sm,color = ['r','b'])
plt.xticks(p,["Non-Smokers","Smokers"])
plt.text(0, 500 ,percentage_of_smokers+'%',color='blue',horizontalalignment='center',verticalalignment='center')
plt.text(1, 500 ,percentage_of_non_smokers+'%',color='red',horizontalalignment='center',verticalalignment='center')
plt.show()
import seaborn as sns
sns.distplot(df[(df.smoker == 1)]["charges"],color='r')
plt.title('Distribution of charges for Smokers')
plt.show()
sns.distplot(df[(df.smoker == 0)]["charges"],color='b')
plt.title('Distribution of charges for Non-Smokers')
plt.show()
df['age'].describe()
columns=['Intervals','count']
df_for_age=pd.DataFrame(0,index=np.arange(7),columns=columns)
df_for_age['Intervals']=df_for_age['Intervals'].astype(str)
n=18
p=25
i=0
while i<7:
    df_for_age['Intervals'][i] = str(n)+'-'+str(p) 
    n=p
    p=7+p
    i=i+1
    
for x in df['age']:
    if(x<=25):
        df_for_age.ix[0,'count']+=1
    elif(x<=32):
         df_for_age.ix[1,'count']+=1
    elif(x<=39):
         df_for_age.ix[2,'count']+=1
    elif(x<=46):
         df_for_age.ix[3,'count']+=1
    elif(x<=53):
         df_for_age.ix[4,'count']+=1
    elif(x<=60):
         df_for_age.ix[5,'count']+=1
    elif(x<=67):
         df_for_age.ix[6,'count']+=1
df_for_age
sns.distplot(df.age)
plt.show()
plt.title("Box plot for charges 18-25 years old smokers")
sns.boxplot(y="smoker", x="charges", data = df[(df.age <= 25 )] , orient="h", palette = 'Set2')

corelation=df.corr()
print(corelation)
sns.heatmap(corelation)
plt.show()
import statsmodels.api as stats

y=df['charges']
X=df[['age','smoker','bmi']]
est=stats.OLS(y,X).fit()
est.summary()

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test =train_test_split(X,y ,test_size=0.20,random_state=42)
dtr=DecisionTreeRegressor(max_depth=4)
dtr.fit(X_train,y_train)
from sklearn.metrics import r2_score,mean_squared_error
print('MSE for training data: ',mean_squared_error(y_train,dtr.predict(X_train)))
print('MSE for testing data: ',mean_squared_error(y_test,dtr.predict(X_test)))
print('R^2 for training data: ',r2_score(y_train, dtr.predict(X_train)))
print('R^2 for testing data: ',r2_score(y_test, dtr.predict(X_test)))
import xgboost
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=4)
xgb.fit(X_train,y_train)
print('MSE for training data: ',mean_squared_error(y_train,xgb.predict(X_train)))
print('MSE for testing data: ',mean_squared_error(y_test,xgb.predict(X_test)))
print('R^2 for training data: ',r2_score(y_train, xgb.predict(X_train)))
print('R^2 for testing data: ',r2_score(y_test, xgb.predict(X_test)))