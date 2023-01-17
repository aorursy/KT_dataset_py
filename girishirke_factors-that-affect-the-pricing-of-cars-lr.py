import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 
import sklearn
from sklearn.model_selection import train_test_split
cars_data = pd.read_csv("../input/car-price-prediction/CarPrice_Assignment.csv")   #Importing the data
cars_data.head()
cars_data.columns   
cars_data.shape  # 205 data-points and 26 variables
cars_data.info()   #No null-values
cars_data.describe()   
# Lets verify the correlation between various variables
plt.figure(figsize=(20,10))
sns.heatmap(cars_data.corr(),annot = True)    
plt.show()
# Dropping the variables
cars_data.drop(['car_ID','carwidth','curbweight','wheelbase','highwaympg'],axis=1,inplace=True)
cars_data['CarName'] = cars_data['CarName'].str.replace('-', ' ')
cars_data['CarName'] = cars_data['CarName'].apply(lambda x : x.split(' ',1)[0])
cars_data['CarName'].unique()
cars_data['CarName'].value_counts()
cars_data['CarName'] = cars_data['CarName'].replace({"toyouta":"toyota","maxda":"mazda","Nissan":"nissan","vw":"volkswagen","vokswagen":"volkswagen","porcshce":"porsche"})
cars_data['CarName'].value_counts()
cars_data.head()
cars_data['fueltype'].value_counts()   #converting into binary variables
cars_data['fueltype'] = cars_data['fueltype'].apply(lambda x : 1 if x=='gas' else 0)
cars_data['fueltype'].value_counts()
cars_data['aspiration'].value_counts()   #converting into binary variables
cars_data['aspiration'] = cars_data['aspiration'].apply(lambda x : 1 if x=='std' else 0)
cars_data['aspiration'].value_counts()
cars_data['doornumber'].value_counts()   #converting into binary variables
cars_data['doornumber'] = cars_data['doornumber'].apply(lambda x : 2 if x=='four' else 1)
cars_data['doornumber'].value_counts()
cars_data['enginelocation'].value_counts()   #converting into binary variables
cars_data['enginelocation'] = cars_data['enginelocation'].apply(lambda x : 1 if x=='front' else 0)
cars_data['enginelocation'].value_counts()
cars_data['cylindernumber'].value_counts()
# Creating dummy variabels for left out categorical variables
cars_data = pd.get_dummies(cars_data)  
cars_data.head()
cars_data.info()
from sklearn.preprocessing import MinMaxScaler  #Lets use min max scaler
scaler = MinMaxScaler()
#Scaling the numeric varibles only
num_vars = ['symboling', 'carlength', 'carheight','enginesize', 'boreratio', 'stroke', 'compressionratio','horsepower', 'peakrpm', 'citympg', 'price']

cars_data[num_vars] = scaler.fit_transform(cars_data[num_vars])


cars_data.describe()
#Spliting the data into train(70%) and test(30%)
from sklearn.model_selection import train_test_split
df_train,df_test = train_test_split(cars_data,train_size=0.7,test_size = 0.3,random_state=100)
y_train = df_train.pop('price')  #Result variable
X_train = df_train               #Predictor variables
from sklearn.feature_selection import RFE 
from sklearn.linear_model import LinearRegression
lm = LinearRegression()          
lm.fit(X_train, y_train)
rfe = RFE(lm, 15)     #Taking 15 variables 
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
rfe_drop = X_train.columns[~rfe.support_]
rfe_drop
X_train = X_train.drop(rfe_drop,axis=1)  #Removing the unwanted variables
X_train.columns
import statsmodels.api as sm        
X_train_rfe_lm = sm.add_constant(X_train)
#First model
lm_1 = sm.OLS(y_train,X_train_rfe_lm).fit()
lm_1.summary()
df_VIF= cars_data.drop(rfe_drop,axis=1)
#Function to find the VIF values of the variables
def vif_cal(input_data, dependent_col):
    vif_df = pd.DataFrame( columns = ['Var', 'Vif'])
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.OLS(y,x).fit().rsquared  
        vif=round(1/(1-rsq),2)
        vif_df.loc[i] = [xvar_names[i], vif]
    return vif_df.sort_values(by = 'Vif', axis=0, ascending=False, inplace=False)
vif_cal(input_data=df_VIF, dependent_col="price")
#Removing 'enginetype_rotor' variable
X_train2 = X_train.drop(['enginetype_rotor'],axis=1)
X_train_rfe_lm2 = sm.add_constant(X_train2)
#Second Model
lm_1 = sm.OLS(y_train,X_train_rfe_lm2).fit()
lm_1.summary()
#Again checking the VIF values for 2nd model
df_VIF = df_VIF.drop('enginetype_rotor', axis =1)
vif_cal(input_data=df_VIF, dependent_col="price")
df_VIF.columns
#Checking the Correlations between all the remaining variables
plt.figure(figsize=(20,10))
sns.heatmap(df_VIF.corr(),annot=True)
plt.show()
df_VIF =df_VIF.drop(['enginesize','boreratio','stroke'],axis=1)
#Lets check the VIF tables again
vif_cal(input_data=df_VIF, dependent_col="price")
X_train2.columns
X_train3 = X_train2.drop(['enginesize','boreratio','stroke'],axis=1)

#Third model
X_train_rfe_lm3 = sm.add_constant(X_train3)
lm_2 = sm.OLS(y_train,X_train_rfe_lm3).fit()
lm_2.summary()

#Making predicitions on training data
y_train_predict = lm_2.predict(X_train_rfe_lm3)
#Plotting error terms
fig = plt.figure()
sns.distplot((y_train - y_train_predict), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

df_test.head()
#Separating result and predictor variables
y_test = df_test.pop('price')
X_test = df_test
X_test.head()
#Adding constant term for statsmodels.api
X_test_new = X_test[X_train3.columns]
X_test_new = sm.add_constant(X_test_new)
X_test_new.head()
#Predicting...
y_test_pred = lm_2.predict(X_test_new)
y_test_pred.head()
y_test.head()
#Finally plotting the predicted y with y_test
fig = plt.figure()
plt.scatter(y_test,y_test_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              
plt.xlabel('y_test', fontsize=18)                          
plt.ylabel('y_pred', fontsize=16) 
from sklearn.metrics import r2_score
r2_score(y_test, y_test_pred)
