#Housing Case Study !!!
import pandas as pd

import numpy as np
data=pd.read_csv('../input/housing-simple-regression/Housing.csv')
data.head()
#What type of values stored in columns...

data.info()
data['furnishingstatus'].value_counts()
#Converting Yes to 1 and No to 0

data['mainroad']=data['mainroad'].map({'yes':1,'no':0})

data['guestroom']=data['guestroom'].map({'yes':1,'no':0})

data['basement']=data['basement'].map({'yes':1,'no':0})

data['hotwaterheating']=data['hotwaterheating'].map({'yes':1,'no':0})

data['airconditioning']=data['airconditioning'].map({'yes':1,'no':0})

data['prefarea']=data['prefarea'].map({'yes':1,'no':0})
data.head()
status=pd.get_dummies(data['furnishingstatus'],drop_first=True)
status.head()
data=pd.concat([data,status],axis=1)
data.head()
data.drop(['furnishingstatus'],axis=1,inplace=True)
data.head()
#CREATING NEW VARIABLES..

data['areaperbedroom']=data['area']/data['bedrooms']

data['bbratio']=data['bathrooms']/data['bedrooms']
data.head()
def normalize(x):

    return ((x-np.min(x))/(max(x)-min(x)))

#Applying Normalize to all columns
data=data.apply(normalize)
data.head()
data.columns
X=data[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',

       'guestroom', 'basement', 'hotwaterheating', 'airconditioning',

       'parking', 'prefarea', 'semi-furnished', 'unfurnished',

       'areaperbedroom', 'bbratio']]

y=data['price']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=100)
import statsmodels.api as sm

X_train=sm.add_constant(X_train)

lm_l=sm.OLS(y_train,X_train).fit()
print(lm_l.summary())
# # Variables might be correlated with each  other...

# # What about if we have more than 1 columns which is highly correlated

# #Correlation matrix can only work for 2 variables but for not more than 2





# We will Use VIF (Variance  Inflation Factor)

# VIF(x)=1/(1-R^2)

#if R^2 Higher VIF also Higher 

#Higher the VIF Higher the muticollinearity   

#Variable with Higher VIF may not be statistically significant

#FUNCTION FOR VIF
def vif_cal(input_data,dependent_col):

    vif_df=pd.DataFrame(columns=['Var','Vif'])

    x_vars=input_data.drop([dependent_col],axis=1)

    x_var_names=x_vars.columns

    for i in range(0,x_var_names.shape[0]):

        y=x_vars[x_var_names[i]]

        x=x_vars[x_var_names.drop(x_var_names[i])]

        rsq=sm.OLS(y,x).fit().rsquared

        vif=round(1/(1-rsq),2)

        vif_df.loc[i]=[x_var_names[i],vif]

    return vif_df.sort_values(by='Vif',axis=0,ascending=False,inplace=False)
vif_cal(input_data=data,dependent_col="price")
# REMOVE BB RATIO BECAUSE IT HAS HIGH P-value and high VIF
#CORRELATION MATRIX
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
plt.figure(figsize=(16,10))

sns.heatmap(data.corr(),annot=True)
#Dropping The variable and Updating The model
X_train=X_train.drop(['bbratio'],axis=1)
lm_2=sm.OLS(y_train,X_train).fit()
print(lm_2.summary())
vif_cal(input_data=data.drop(['bbratio'],axis=1),dependent_col="price")
#DROPPING BEDROOM ALSO

X_train=X_train.drop(['bedrooms'],axis=1)
lm_3=sm.OLS(y_train,X_train).fit()
print(lm_3.summary())
vif_cal(input_data=data.drop(['bbratio','bedrooms'],axis=1),dependent_col="price")
#Removing Area PEr Bedroom
X_train=X_train.drop(['areaperbedroom'],axis=1)
lm_4=sm.OLS(y_train,X_train).fit()
lm_4.summary()
vif_cal(input_data=data.drop(['bbratio','bedrooms','areaperbedroom'],axis=1),dependent_col="price")
#REMOVING SEMI-FURNISHED ALSO
X_train.drop(['semi-furnished','basement'],axis=1,inplace=True)
lm_5=sm.OLS(y_train,X_train).fit()
print(lm_5.summary())
vif_cal(input_data=data.drop(['bbratio','bedrooms','areaperbedroom','semi-furnished','basement'],axis=1),dependent_col="price")
# DESIGN model by dropping all variables one by one with high VIF. Then compare results
X_test_m5=sm.add_constant(X_test)
X_test_m5=X_test_m5.drop(['bbratio','bedrooms','areaperbedroom','semi-furnished','basement'],axis=1)
y_pred_m5=lm_5.predict(X_test_m5)
#MODEL EVALUATION
from sklearn.metrics import mean_squared_error,r2_score

mse=mean_squared_error(y_test,y_pred_m5)

rsq=r2_score(y_test,y_pred_m5)
mse
rsq
#We can also use RFE...