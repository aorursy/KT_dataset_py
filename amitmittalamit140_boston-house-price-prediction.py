#importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings #to remove warning from the notebook

warnings.filterwarnings(action='ignore')
#loading dataset

name= ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

df = pd.read_csv(filepath_or_buffer="../input/boston-house-prices/housing.csv",delim_whitespace=True,names=name)

df.head()
#shape of our dataset

df.shape
#information about the data

df.info()
#checking for missing data

df.isnull().sum()

#there is no missing value in the data
plt.figure(figsize=(12,12))

sns.heatmap(data=df.corr().round(2),annot=True,cmap='coolwarm',linewidths=0.2,square=True)
#since some of these features shows quite good and very good correlation with our predictive variable Houese Price(MEDV)

df1 = df[['RM','TAX','PTRATIO','LSTAT','MEDV']]

df1.head()
sns.pairplot(data=df1)
#description about data

desc = df1.describe().round(2)

desc
#Box Plot and Distribution Plot for Dependent variable MEDV

plt.figure(figsize=(20,3))



plt.subplot(1,2,1)

sns.boxplot(df1.MEDV,color='#005030')

plt.title('Box Plot of MEDV')



plt.subplot(1,2,2)

sns.distplot(a=df1.MEDV,color='#500050')

plt.title('Distribution Plot of MEDV')

plt.show()
MEDV_Q3 = desc['MEDV']['75%']

MEDV_Q1 = desc['MEDV']['25%']

MEDV_IQR = MEDV_Q3 - MEDV_Q1

MEDV_UV = MEDV_Q3 + 1.5*MEDV_IQR

MEDV_LV = MEDV_Q1 - 1.5*MEDV_IQR



df1[df1['MEDV']<MEDV_LV]
df1[df1['MEDV']>MEDV_UV].sort_values(by=['MEDV','RM'])
print(f'Shape of dataset before remving Outliers: {df1.shape}')

df2 = df1[~(df1['MEDV']==50)]

print(f'Shape of dataset after remving Outliers: {df2.shape}')
#Box Plot, Distribution Plot and Scatter Plot for TAX

plt.figure(figsize=(20,3))



plt.subplot(1,3,1)

sns.boxplot(df2.TAX,color='#005030')

plt.title('Box Plot of TAX')



plt.subplot(1,3,2)

sns.distplot(a=df2.TAX,color='#500050')

plt.title('Distribution Plot of TAX')



plt.subplot(1,3,3)

sns.scatterplot(df2.TAX,df2.MEDV)

plt.title('Scatter Plot of TAX vs MEDV')



plt.show()
temp_df = df2[df1['TAX']>600].sort_values(by=['RM','MEDV'])

temp_df.shape
temp_df
temp_df.describe()
TAX_10 = df2[(df2['TAX']<600) & (df2['LSTAT']>=0) & (df2['LSTAT']<10)]['TAX'].mean()

TAX_20 = df2[(df2['TAX']<600) & (df2['LSTAT']>=10) & (df2['LSTAT']<20)]['TAX'].mean()

TAX_30 = df2[(df2['TAX']<600) & (df2['LSTAT']>=20) & (df2['LSTAT']<30)]['TAX'].mean()

TAX_40 = df2[(df2['TAX']<600) & (df2['LSTAT']>=30)]['TAX'].mean()



indexes = list(df2.index)

for i in indexes:

    if df2['TAX'][i] > 600:

        if (0 <= df2['LSTAT'][i] < 10):

            df2.at[i,'TAX'] = TAX_10

        elif (10 <= df2['LSTAT'][i] < 20):

            df2.at[i,'TAX'] = TAX_20

        elif (20 <= df2['LSTAT'][i] < 30):

            df2.at[i,'TAX'] = TAX_30

        elif (df2['LSTAT'][i] >30):

            df2.at[i,'TAX'] = TAX_40



print('Values imputed successfully')
#This show all those extreme TAX values are replaced successfully

df2[df2['TAX']>600]['TAX'].count()
sns.distplot(a=df2.TAX,color='#500050')

plt.title('Distribution Plot of TAX after replacing extreme values')

plt.show()
#Box Plot, Distribution Plot and Scatter Plot for PTRATIO

plt.figure(figsize=(20,3))



plt.subplot(1,3,1)

sns.boxplot(df2.PTRATIO,color='#005030')

plt.title('Box Plot of PTRATIO')



plt.subplot(1,3,2)

sns.distplot(a=df2.PTRATIO,color='#500050')

plt.title('Distribution Plot of PTRATIO')



plt.subplot(1,3,3)

sns.scatterplot(df2.PTRATIO,df2.MEDV)

plt.title('Scatter Plot of PTRATIO vs MEDV')



plt.show()
df2[df2['PTRATIO']<14].sort_values(by=['LSTAT','MEDV'])
#Box Plot, Distribution Plot and Scatter Plot for LSTAT

plt.figure(figsize=(20,3))



plt.subplot(1,3,1)

sns.boxplot(df2.LSTAT,color='#005030')

plt.title('Box Plot of LSTAT')



plt.subplot(1,3,2)

sns.distplot(a=df2.LSTAT,color='#500050')

plt.title('Distribution Plot of LSTAT')



plt.subplot(1,3,3)

sns.scatterplot(df2.LSTAT,df2.MEDV)

plt.title('Scatter Plot of LSTAT vs MEDV')



plt.show()
LSTAT_Q3 = desc['LSTAT']['75%']

LSTAT_Q1 = desc['LSTAT']['25%']

LSTAT_IQR = LSTAT_Q3 - LSTAT_Q1

LSTAT_UV = LSTAT_Q3 + 1.5*LSTAT_IQR

LSTAT_LV = LSTAT_Q1 - 1.5*LSTAT_IQR



df2[df2['LSTAT']>LSTAT_UV].sort_values(by='LSTAT')
#Box Plot, Distribution Plot and Scatter Plot for RM

plt.figure(figsize=(20,3))



plt.subplot(1,3,1)

sns.boxplot(df2.RM,color='#005030')

plt.title('Box Plot of MEDV')



plt.subplot(1,3,2)

sns.distplot(a=df2.RM,color='#500050')

plt.title('Distribution Plot of MEDV')



plt.subplot(1,3,3)

sns.scatterplot(df2.RM,df2.MEDV)

plt.title('Scatter Plot of RM vs MEDV')



plt.show()
RM_Q3 = desc['RM']['75%']

RM_Q1 = desc['RM']['25%']

RM_IQR = RM_Q3 - RM_Q1

RM_UV = RM_Q3 + 1.5*RM_IQR

RM_LV = RM_Q1 - 1.5*RM_IQR



df2[df2['RM']<RM_LV].sort_values(by=['RM','MEDV'])
print(f'Shape of dataset before removing data points: {df2.shape}')

df3 = df2.drop(axis=0,index=[365,367])

print(f'Shape of dataset before removing data points: {df3.shape}')
df3[df3['RM']>RM_UV].sort_values(by=['RM','MEDV'])
print(f'Shape of dataset before removing data points: {df3.shape}')

df3 = df3.drop(axis=0,index=[364])

print(f'Shape of dataset before removing data points: {df3.shape}')
#Now will split our dataset into Dependent variable and Independent variable



X = df3.iloc[:,0:4].values

y = df3.iloc[:,-1:].values
print(f"Shape of Dependent Variable X = {X.shape}")

print(f"Shape of Independent Variable y = {y.shape}")
def FeatureScaling(X):

    """

    is function takes an array as an input, which needs to be scaled down.

    Apply Standardization technique to it and scale down the features with mean = 0 and standard deviation = 1

    

    Input <- 2 dimensional numpy array

    Returns -> Numpy array after applying Feature Scaling

    """

    mean = np.mean(X,axis=0)

    std = np.std(X,axis=0)

    for i in range(X.shape[1]):

        X[:,i] = (X[:,i]-mean[i])/std[i]



    return X
X = FeatureScaling(X)
m,n = X.shape

X = np.append(arr=np.ones((m,1)),values=X,axis=1)
#Now we will spit our data into Train set and Test Set



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 42)



print(f"Shape of X_train = {X_train.shape}")

print(f"Shape of X_test = {X_test.shape}")

print(f"Shape of y_train = {y_train.shape}")

print(f"Shape of y_test = {y_test.shape}")
#ComputeCost function determines the cost (sum of squared errors) 



def ComputeCost(X,y,theta):

    """

    This function takes three inputs and uses the Cost Function to determine the cost (basically error of prediction vs

    actual values)

    Cost Function: Sum of square of error in predicted values divided by number of data points in the set

    J = 1/(2*m) *  Summation(Square(Predicted values - Actual values))

    

    Input <- Take three numoy array X,y and theta

    Return -> The cost calculated from the Cost Function

    """

    m=X.shape[0] #number of data points in the set

    J = (1/(2*m)) * np.sum((X.dot(theta) - y)**2)

    return J
#Gradient Descent Algorithm to minimize the Cost and find best parameters in order to get best line for our dataset



def GradientDescent(X,y,theta,alpha,no_of_iters):

    """

    Gradient Descent Algorithm to minimize the Cost

    

    Input <- X, y and theta are numpy arrays

            X -> Independent Variables/ Features

            y -> Dependent/ Target Variable

            theta -> Parameters 

            alpha -> Learning Rate i.e. size of each steps we take

            no_of_iters -> Number of iterations we want to perform

    

    Return -> theta (numpy array) which are the best parameters for our dataset to fit a linear line

             and Cost Computed (numpy array) for each iteration

    """

    m=X.shape[0]

    J_Cost = []

    for i in range(no_of_iters):

        error = np.dot(X.transpose(),(X.dot(theta)-y))

        theta = theta - alpha * (1/m) * error

        J_Cost.append(ComputeCost(X,y,theta))

    

    return theta, np.array(J_Cost)
iters = 1000



alpha1 = 0.001

theta1 = np.zeros((X_train.shape[1],1))

theta1, J_Costs1 = GradientDescent(X_train,y_train,theta1,alpha1,iters)



alpha2 = 0.003

theta2 = np.zeros((X_train.shape[1],1))

theta2, J_Costs2 = GradientDescent(X_train,y_train,theta2,alpha2,iters)



alpha3 = 0.01

theta3 = np.zeros((X_train.shape[1],1))

theta3, J_Costs3 = GradientDescent(X_train,y_train,theta3,alpha3,iters)



alpha4 = 0.03

theta4 = np.zeros((X_train.shape[1],1))

theta4, J_Costs4 = GradientDescent(X_train,y_train,theta4,alpha4,iters)
plt.figure(figsize=(8,5))

plt.plot(J_Costs1,label = 'alpha = 0.001')

plt.plot(J_Costs2,label = 'alpha = 0.003')

plt.plot(J_Costs3,label = 'alpha = 0.01')

plt.plot(J_Costs4,label = 'alpha = 0.03')

plt.title('Convergence of Gradient Descent for different values of alpha')

plt.xlabel('No. of iterations')

plt.ylabel('Cost')

plt.legend()

plt.show()
theta4
def Predict(X,theta):

    """

    This function predicts the result for the unseen data

    """

    y_pred = X.dot(theta)

    return y_pred
y_pred = Predict(X_test,theta4)

y_pred[:5]
plt.scatter(x=y_test,y=y_pred,alpha=0.5)

plt.xlabel('y_test',size=12)

plt.ylabel('y_pred',size=12)

plt.title('Predicited Values vs Original Values (Test Set)',size=15)

plt.show()
sns.residplot(y_pred,(y_pred-y_test))

plt.xlabel('Predicited Values',size=12)

plt.ylabel("Residues",size=12)

plt.title('Residual Plot',size=15)

plt.show()
sns.distplot(y_pred-y_test)

plt.xlabel('Residual',size=12)

plt.ylabel('Frquency',size=12)

plt.title('Distribution of Residuals',size=15)

plt.show()
from sklearn import metrics

r2= metrics.r2_score(y_test,y_pred)

N,p = X_test.shape

adj_r2 = 1-((1-r2)*(N-1))/(N-p-1)

print(f'R^2 = {r2}')

print(f'Adjusted R^2 = {adj_r2}')
from sklearn import metrics

mse = metrics.mean_squared_error(y_test,y_pred)

mae = metrics.mean_absolute_error(y_test,y_pred)

rmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))

print(f'Mean Squared Error: {mse}',f'Mean Absolute Error: {mae}',f'Root Mean Squared Error: {rmse}',sep='\n')
#coefficients of regression model

coeff=np.array([y for x in theta4 for y in x]).round(2)

features=['Bias','RM','TAX','PTRATIO','LSTAT']

eqn = 'MEDV = '

for f,c in zip(features,coeff):

    eqn+=f" + ({c} * {f})";



print(eqn)
sns.barplot(x=features,y=coeff)

plt.ylim([-5,25])

plt.xlabel('Coefficient Names',size=12)

plt.ylabel('Coefficient Values',size=12)

plt.title('Visualising Regression Coefficients',size=15)

plt.show()
X_dt = df3.iloc[:,:-1].values

y_dt = df3.iloc[:,-1].values
from sklearn.model_selection import train_test_split

X_train_dt,X_test_dt,y_train_dt,y_test_dt = train_test_split(X_dt,y_dt,test_size=0.2,random_state=42)



print(f"Shape of X_train_dt = {X_train_dt.shape}")

print(f"Shape of X_test_dt = {X_test_dt.shape}")

print(f"Shape of y_train_dt = {y_train_dt.shape}")

print(f"Shape of y_test_dt = {y_test_dt.shape}")
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()

dt.fit(X_train_dt,y_train_dt)
y_pred_dt = dt.predict(X_test_dt)

y_pred_dt[:5]
plt.scatter(x=y_test_dt,y=y_pred_dt,alpha=0.5)

plt.xlabel('y_test',size=12)

plt.ylabel('y_pred',size=12)

plt.title('Predicited Values vs Original Values (Test Set)',size=15)

plt.show()
sns.residplot(y_pred_dt,(y_pred_dt-y_test_dt))

plt.xlabel('Predicited Values',size=12)

plt.ylabel("Residues",size=12)

plt.title('Residual Plot',size=15)

plt.show()
sns.distplot(y_pred_dt-y_test_dt)

plt.xlabel('Residual',size=12)

plt.ylabel('Frquency',size=12)

plt.title('Distribution of Residuals',size=15)

plt.show()
from sklearn import metrics

r2_dt= metrics.r2_score(y_test_dt,y_pred_dt)

N,p = X_test_dt.shape

adj_r2_dt = 1-((1-r2_dt)*(N-1))/(N-p-1)

print(f'R^2 = {r2_dt}')

print(f'Adjusted R^2 = {adj_r2_dt}')
from sklearn import metrics

mse_dt = metrics.mean_squared_error(y_test_dt,y_pred_dt)

mae_dt = metrics.mean_absolute_error(y_test_dt,y_pred_dt)

rmse_dt = np.sqrt(metrics.mean_squared_error(y_test_dt,y_pred_dt))

print(f'Mean Squared Error: {mse_dt}',f'Mean Absolute Error: {mae_dt}',f'Root Mean Squared Error: {rmse_dt}',sep='\n')
X_rf = df3.iloc[:,:-1].values

y_rf = df3.iloc[:,-1].values
from sklearn.model_selection import train_test_split

X_train_rf,X_test_rf,y_train_rf,y_test_rf = train_test_split(X_rf,y_rf,test_size=0.2,random_state=42)



print(f"Shape of X_train_rf = {X_train_rf.shape}")

print(f"Shape of X_test_rf = {X_test_rf.shape}")

print(f"Shape of y_train_rf = {y_train_rf.shape}")

print(f"Shape of y_test_rf = {y_test_rf.shape}")
warnings.filterwarnings(action='ignore')

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42)

rf.fit(X_train_rf,y_train_rf)
y_pred_rf = rf.predict(X_test_rf)

y_pred_rf[:5]
plt.scatter(x=y_test_rf,y=y_pred_rf,alpha=0.5)

plt.xlabel('y_test',size=12)

plt.ylabel('y_pred',size=12)

plt.title('Predicited Values vs Original Values (Test Set)',size=15)

plt.show()
sns.residplot(y_pred_rf,(y_pred_rf-y_test_rf))

plt.xlabel('Predicited Values',size=12)

plt.ylabel("Residues",size=12)

plt.title('Residual Plot',size=15)

plt.show()
sns.distplot(y_pred_rf-y_test_rf)

plt.xlabel('Residual',size=12)

plt.ylabel('Frquency',size=12)

plt.title('Distribution of Residuals',size=15)

plt.show()
from sklearn import metrics

r2_rf= metrics.r2_score(y_test_rf,y_pred_rf)

N,p = X_test_dt.shape

adj_r2_rf = 1-((1-r2_rf)*(N-1))/(N-p-1)

print(f'R^2 = {r2_rf}')

print(f'Adjusted R^2 = {adj_r2_rf}')

from sklearn import metrics

mse_rf = metrics.mean_squared_error(y_test_rf,y_pred_rf)

mae_rf = metrics.mean_absolute_error(y_test_rf,y_pred_rf)

rmse_rf = np.sqrt(metrics.mean_squared_error(y_test_rf,y_pred_rf))

print(f'Mean Squared Error: {mse_rf}',f'Mean Absolute Error: {mae_rf}',f'Root Mean Squared Error: {rmse_rf}',sep='\n')
results=pd.DataFrame({'Linear Regression':[r2,adj_r2],'Decision Tree':[r2_dt,adj_r2_dt],

                      'Random Forest':[r2_rf,adj_r2_rf]},index=['R square','Adj R square'])



results.plot(kind='bar',alpha=0.7,grid=True,title='Interpreting Results',rot=0,figsize=(10,5),colormap='jet')

results