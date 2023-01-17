import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn import preprocessing,metrics

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.feature_selection import RFE

import statsmodels.api as sm

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")

df=pd.read_csv('../input/predictive-analysis/CarPrice_Assignment.csv')

df.head()

## DIVIDING DATA BETWEEN TEST AND TRAIN

# We specify this so that the train and test data set always have the same rows, respectively

df, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 8)
df.CarName=df.CarName.str.lower()

df_test.CarName=df_test.CarName.str.lower()

df[['Company_Name','Car_Name']]= df.CarName.str.split(' ',n=1,expand=True)

df_test[['Company_Name','Car_Name']]= df_test.CarName.str.split(' ',n=1,expand=True)

### MAKING CORRECTIONS TO COMPANY NAME

#df[(df.Company_Name=='maxda') | (df.Company_Name=='mazda')] # Only TWO ENTRIES WITH COMPANY NAME MAXDA

#since 'x' and 'z' are next to each other on a standard QWERTY keyboard could have been entry error

df.Company_Name=df.Company_Name.apply(lambda x: 'mazda' if(str(x)=='maxda') else x)

df.Company_Name=df.Company_Name.apply(lambda x: 'porsche' if(str(x)=='porcshce') else x)

df.Company_Name=df.Company_Name.apply(lambda x: 'toyota' if(str(x)=='toyouta') else x)

df.Company_Name=df.Company_Name.apply(lambda x: 'volkswagen' if(str(x)=='vokswagen' or str(x)=='vw') else x)



df_test.Company_Name=df_test.Company_Name.apply(lambda x: 'mazda' if(str(x)=='maxda') else x)

df_test.Company_Name=df_test.Company_Name.apply(lambda x: 'porsche' if(str(x)=='porcshce') else x)

df_test.Company_Name=df_test.Company_Name.apply(lambda x: 'toyota' if(str(x)=='toyouta') else x)

df_test.Company_Name=df_test.Company_Name.apply(lambda x: 'volkswagen' if(str(x)=='vokswagen' or str(x)=='vw') else x)



df=df.drop(['CarName','Car_Name'],axis=1)

df_test=df_test.drop(['CarName','Car_Name'],axis=1)

fig = plt.figure(figsize=(15, 10))

ax=sns.countplot(df['Company_Name'])

plt.xticks(rotation='vertical')

plt.xlabel('Company_Name')

plt.ylabel('Cars Sold')

plt.title('Sales Count')

#plt.yticks(rotation='vertical')

plt.show()
## LOOKING AT PRICE DISTRIBUTION FROM TRAINING DATA

plt.rcParams['figure.figsize']=(23,10)

ax = sns.boxplot(x="Company_Name", y="price", data=df)

ax.set_xlabel('Company_Name')

ax.set_ylabel('Price in $')

plt.title('Price distribution Company wise')



## Inference: More cars use gas as a fuel compared to diesel

f, axis = plt.subplots(figsize=(7, 7))

sns.countplot(x='fueltype', data=df)

plt.title("Fuel type frequency ")

plt.ylabel('Number of vehicles')

plt.xlabel('Fuel type')
sns.boxplot(x="aspiration", y="price", data=df)

## As expected cars with turbo aspiration are priced higher. But there are some cars with 

## standard aspiration which are priced higher than the turbo ones. Outliers? Maybe not. 

sns.regplot(x="horsepower", y="price", data=df)

## Higher the horse power higher the cost. Heteroscedastic linear relationship.
df['horsepower'].hist()

plt.title("Horsepower histogram")

plt.ylabel('Number of vehicles')

plt.xlabel('Horsepower')

## More number of cars with lower horse power.
f, axis = plt.subplots(figsize=(7, 7))

sns.countplot(x='drivewheel', data=df)

plt.title("Drivewheel frequency ")

plt.ylabel('Number of vehicles')

plt.xlabel('Drivewheel type')

print(df.groupby('drivewheel').mean()['price'])

print(df.groupby('drivewheel').count()['car_ID'])
##MPFI AND MFI ARE THE SAME THING: Multipoint fuel injection

##SPDI seems to be just a spelling mistake for SPFI.

df['fuelsystem']=df['fuelsystem'].apply(lambda x: 'mpfi' if(str(x)=='mfi') else str(x))

df['fuelsystem']=df['fuelsystem'].apply(lambda x: 'spfi' if(str(x)=='spdi') else str(x))

df_test['fuelsystem']=df_test['fuelsystem'].apply(lambda x: 'mpfi' if(str(x)=='mfi') else str(x))

df_test['fuelsystem']=df_test['fuelsystem'].apply(lambda x: 'spfi' if(str(x)=='spdi') else str(x))

f, axis = plt.subplots(figsize=(7, 7))

sns.countplot(x='fuelsystem', data=df)

plt.title("Fuel type frequency ")

plt.ylabel('Number of vehicles')

plt.xlabel('Fuel System Type')

print(df.groupby('fuelsystem').mean()['price'].sort_values(ascending=False))
f, axis = plt.subplots(figsize=(7, 7))

sns.countplot(x='carbody', data=df)

plt.title("carbody ")

plt.ylabel('Number of vehicles')

plt.xlabel('carbody')

print(df.carbody.describe())

print(df.groupby('carbody').mean()['price'])
f, axis = plt.subplots(figsize=(7, 7))

sns.countplot(x='enginelocation', data=df)

plt.title("enginelocation ")

plt.ylabel('Number of vehicles')

plt.xlabel('enginelocation')

df.enginelocation.describe()

# Highly biased column
f, axis = plt.subplots(figsize=(7, 7))

sns.countplot(x='cylindernumber', data=df)

plt.title("Number of Cylinder frequency ")

plt.ylabel('Number of vehicles')

plt.xlabel('Number of Cylinders')

df.cylindernumber.describe()



print(df.enginetype.describe())

## OTHER ENGINETYPES WOULD JUST ADD COMPLEXITY TO THE MODEL

df['enginetype']=df['enginetype'].apply(lambda x: str(x) if(str(x)=='ohc') else 'others')

df_test['enginetype']=df_test['enginetype'].apply(lambda x: str(x) if(str(x)=='ohc') else 'others')

f, axis = plt.subplots(figsize=(7, 7))

sns.countplot(x='enginetype', data=df)

plt.title("Fuel type frequency ")

plt.ylabel('Number of vehicles')

plt.xlabel('Engine Type')
sns.lmplot("enginesize",'price', df)

plt.title("Price Vs Engine-size")

##Inference: Higher the engine-size is, costlier the vehicle is.
## NUMERICAL DATA ENGINEERING: checking on Training data

## SINCE NAME OF CAR MANUFACTURER IS OF NO USE FOR OUR REGRESSION ANALYSIS

## WE WILL REPRESENT THE CA MANUFACTURER BY MEDIAN COSTS OF THEIR CARS

CompanyInfo=pd.DataFrame(df.groupby('Company_Name')['price'].median()).reset_index()

CompanyInfo.rename(columns={'price':'price_median'},inplace=True)

CompanyInfo.head()
plt.figure(figsize=(20,10))

corr_matrix=df.corr()

sns.heatmap(corr_matrix,annot=True).set_title('Correlation for Training Data')
df.groupby(['Company_Name','enginesize'])['carbody','price'].mean()[:10]

# Every company has a set of standard engine-sizes.
sns.boxplot(x="Company_Name", y="enginesize", data=df)
train_unique=[]

train_columns=[]

train_categorical=[]

train_numerical=[]

def categorical2binary(train_unique,train_column,col,df):

    df[col]=df[col].apply(lambda x: list(train_unique[train_column.index(col)]).index(str(x)))

    return df[col]

for i in df.columns:

    if(df[i].dtype==object):

        train_categorical.append(i)

        train_columns.append(i)

        train_unique.append(df[i].unique())

        if(len(df[i].unique())==2): ## ENCODING COLUMNS WITH 2 UNIQUE VALUES TO BINARY

            df[i]=categorical2binary(train_unique,train_columns,i,df)

            ##ENCODING TEST DATA USING METRIC TRAIN DATA

            df_test[i]=categorical2binary(train_unique,train_columns,i,df_test)

            train_categorical.remove(i)

    else:

        train_numerical.append(i)

print("Categorical Columns: ",train_categorical)

print("Numerical Columns: ",train_numerical)

df_test.head()
## MERGING TEST AND TRAIN TO PERFORM DUMMY ENCODING

train_length = len(df.index)

df_merged = pd.concat(objs=[df,df_test], axis=0)

dummy_df_merged = pd.get_dummies(df_merged[train_categorical])#COnvert categorical columns 

#to dummy columns except 'Company_Name' as it has been replaced by 'Company_average_price'

# for analysis

df_merged=df_merged.drop(train_categorical,axis=1) ## Remove all categorical columns 

#after encoding

## Dropping Car_ID from both train and test data

df_merged=df_merged.drop('car_ID',axis=1)

train_numerical.remove('car_ID')
## Rearranging columns to make dependent variable the last column

df_merged = pd.concat([df_merged,dummy_df_merged], axis = 1)

columnrearrange=list(df_merged.columns)

columnrearrange.remove('price')

columnrearrange.append('price')

train_numerical.remove('price')

df = df_merged[:train_length]

df_test = df_merged[train_length:]

df=df[columnrearrange]

df_test=df_test[columnrearrange]

df_test.head()
sns.pairplot(df.corr(),y_vars='price',x_vars=df.columns).fig.suptitle('Dependent and Independent variable correlation')
## DIVIDING TRAINING DATA AND TESTING DATA

X_train=df[df.columns[:-1]].copy()

y_train=df[df.columns[-1:]].copy()

X_test=df_test[df_test.columns[:-1]].copy()

y_test=df_test[df_test.columns[-1:]].copy()



## Scaling train and test data : StandardScaler

normalize = preprocessing.StandardScaler()

X_train[train_numerical]=normalize.fit_transform(X_train[train_numerical])

X_test[train_numerical]=normalize.transform(X_test[train_numerical])

X_train = pd.DataFrame(X_train,columns=X_train.columns)

X_test=pd.DataFrame(X_test,columns=X_test.columns)



# X_test.columns,y_test.columns

X_train.head()



col = X_train.columns

X_train_1 = X_train[col].copy()

X_train_1 = sm.add_constant(X_train_1)

lm1= sm.OLS(np.array(y_train),X_train_1).fit()   # Fitting the linear model

X_test_1= X_test[col].copy()

X_test_1= sm.add_constant(X_test_1,has_constant='add')

y_pred_1=lm1.predict(X_test_1)

mse1 = mean_squared_error(y_test,y_pred_1)

print("Mean Squared Error: ",mse1)

print("R2_score on test data: ",r2_score(y_test,y_pred_1)," \n")

print("Model Summary\n")

print(lm1.summary())

## R2= 0.927 : Adjusted_R2= 0.903 -> Seems like we are facing a high penalty


lm2 = LinearRegression()

lm2.fit(X_train, y_train)

rfe = RFE(lm2,25) 

rfe = rfe.fit(X_train, y_train)

col = X_train.columns[rfe.support_]



## COlumns with ranking  1 will be used for futher analysis
X_train_rfe_1 = X_train[col].copy()

X_train_rfe_1 = sm.add_constant(X_train_rfe_1)

lm2 = sm.OLS(np.array(y_train),X_train_rfe_1).fit()   # Running the linear model

X_test_2 = X_test[col].copy()

X_test_2 = sm.add_constant(X_test_2, has_constant='add')

y_pred_2 = lm2.predict(X_test_2)

mse2 = mean_squared_error(y_test,y_pred_2)

print("Columns Used: ",len(list(X_train_rfe_1)))

print("Mean Squared Error: ",mse2)

print("R2_score on test data: ",r2_score(y_test,y_pred_2)," \n")

############################################################

## MEAN SQUARED ERROR IMPROVEMENT INDICATES THE DECREASE IN ERROR COMPARED TO FIRST MODEL

## NEGATIVE WOULD MEAN THE ERROR INCREASED

## POSITIVE WOULD MEAN THE ERROR DECREASED

print("Mean Squared Error Improvement= ",mse1-mse2)

############################################################

print("Model Summary\n")

print(lm2.summary())

## Although we have reduced the amount of penalty on the model but 

## we have also lost on some R_squared score. 
## Just by looking at the data we can see that a lot of values have high p value

## Onto feature filtering using p-values and vif
summary_df=pd.DataFrame({"p":round(lm2.pvalues,3)}).reset_index()

# Calculate the VIFs for the new model

vif = pd.DataFrame()

vif['index'] = X_train_rfe_1.columns

vif['VIF'] = [variance_inflation_factor(X_train_rfe_1.values, i) for i in range(X_train_rfe_1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)



vif = vif.sort_values(by = "VIF", ascending = False)

vif.head()

summary_df=pd.merge(summary_df,vif,on='index')

summary_df=summary_df.sort_values(['p','VIF'],ascending=[0,0])

summary_df.head(10)
col=list(X_train_rfe_1.columns)

col.remove('cylindernumber_six')

X_train_2 = X_train_rfe_1[col].copy()

X_train_2 = sm.add_constant(X_train_2) #-> Won't add column 

lm3 = sm.OLS(np.array(y_train),X_train_2).fit()   # Running the linear model

X_test_3 = X_test_2[col].copy()

X_test_3 = sm.add_constant(X_test_3)

y_pred_3 = lm3.predict(X_test_3)

mse3 = mean_squared_error(y_test,y_pred_3)

print("Columns Used: ",len(list(col)))

print("Mean Squared Error: ",mse3,)

print("Mean Squared Error Improvement= ",mse1-mse3)

print("R2_score on test data: ",r2_score(y_test,y_pred_3)," \n")

print("Model Summary\n")

print(lm3.summary())

summary_df=pd.DataFrame({"p":round(lm3.pvalues,3)}).reset_index()

summary_df.head()

# Calculate the VIFs for the new model

vif = pd.DataFrame()

vif['index'] = X_train_2.columns

vif['VIF'] = [variance_inflation_factor(X_train_2.values, i) for i in range(X_train_2.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif.head()

summary_df=pd.merge(summary_df,vif,on='index')

summary_df=summary_df.sort_values(['p','VIF'],ascending=[0,0])

summary_df.head()
col=list(X_train_2.columns)

col.remove('cylindernumber_eight')

X_train_3 = X_train_2[col].copy()

X_train_3 = sm.add_constant(X_train_3)#> Won't add column 

lm4 = sm.OLS(np.array(y_train),X_train_3).fit()   # Running the linear model

X_test_4 = X_test_3[col].copy()

X_test_4 = sm.add_constant(X_test_4)

y_pred_4 = lm4.predict(X_test_4)

mse4 = mean_squared_error(y_test,y_pred_4)

print("Columns Used: ",len(list(col)))

print("Mean Squared Error: ",mse4,)

print("Mean Squared Error Improvement= ",mse1-mse4)

print("R2_score on test data: ",r2_score(y_test,y_pred_4)," \n")

print("Model Summary\n")

print(lm4.summary())

summary_df=pd.DataFrame({"p":round(lm4.pvalues,3)}).reset_index()

summary_df.head()

# Calculate the VIFs for the new model

vif = pd.DataFrame()

vif['index'] = X_train_3.columns

vif['VIF'] = [variance_inflation_factor(X_train_3.values, i) for i in range(X_train_3.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif.head()

## SINCE ALL P-VALUES ARE NOW BELOW 0.05 WE WILL BE REMOVING COLUMNS WITH HIGH VIF

summary_df=pd.merge(summary_df,vif,on='index')

summary_df=summary_df.sort_values(['p','VIF'],ascending=[0,0])

summary_df.head()
col=list(X_train_3.columns)

col.remove('Company_Name_chevrolet')

X_train_4 = X_train_3[col].copy()

X_train_4 = sm.add_constant(X_train_4)

lm5 = sm.OLS(np.array(y_train),X_train_4).fit()   # Running the linear model

X_test_5 = X_test_4[col].copy()

X_test_5 = sm.add_constant(X_test_5)

y_pred_5 = lm5.predict(X_test_5)

mse5 = mean_squared_error(y_test,y_pred_5)

print("Columns Used: ",len(list(col)))

print("Mean Squared Error: ",mse5,)

print("Mean Squared Error Improvement= ",mse1-mse5)

print("R2_score on test data: ",r2_score(y_test,y_pred_5)," \n")

print("Model Summary\n")

print(lm5.summary())

summary_df=pd.DataFrame({"p":round(lm5.pvalues,3)}).reset_index()

summary_df.head()

# Calculate the VIFs for the new model

vif = pd.DataFrame()

vif['index'] = X_train_4.columns

vif['VIF'] = [variance_inflation_factor(X_train_4.values, i) for i in range(X_train_4.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif.head()

## SINCE ALL P-VALUES ARE NOW BELOW 0.05 WE WILL BE REMOVING COLUMNS WITH HIGH VIF

summary_df=pd.merge(summary_df,vif,on='index')

summary_df=summary_df.sort_values(['p','VIF'],ascending=[0,0])

summary_df.head()
col=list(X_train_4.columns)

col.remove('Company_Name_mercury')

X_train_5 = X_train_4[col].copy()

X_train_5 = sm.add_constant(X_train_5)

lm6 = sm.OLS(np.array(y_train),X_train_5).fit()   # Running the linear model

X_test_6 = X_test_5[col].copy()

X_test_6 = sm.add_constant(X_test_6)

y_pred_6 = lm6.predict(X_test_6)

mse6 = mean_squared_error(y_test,y_pred_6)

print("Columns Used: ",len(list(col)))

print("Mean Squared Error: ",mse6,)

print("Mean Squared Error Improvement= ",mse1-mse6)

print("R2_score on test data: ",r2_score(y_test,y_pred_6)," \n")

print("Model Summary\n")

print(lm6.summary())

summary_df=pd.DataFrame({"p":round(lm6.pvalues,3)}).reset_index()

summary_df.head()

# Calculate the VIFs for the new model

vif = pd.DataFrame()

vif['index'] = X_train_5.columns

vif['VIF'] = [variance_inflation_factor(X_train_5.values, i) for i in range(X_train_5.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif.head()

## SINCE ALL P-VALUES ARE NOW BELOW 0.05 WE WILL BE REMOVING COLUMNS WITH HIGH VIF

summary_df=pd.merge(summary_df,vif,on='index')

summary_df=summary_df.sort_values(['p','VIF'],ascending=[0,0])

summary_df.head()
col=list(X_train_5.columns)

col.remove('fuelsystem_4bbl')

X_train_6 = X_train_5[col].copy()

X_train_6 = sm.add_constant(X_train_6) #-> Won't add column 

lm7 = sm.OLS(np.array(y_train),X_train_6).fit()   # Running the linear model

X_test_7 = X_test_6[col].copy()

X_test_7 = sm.add_constant(X_test_7)

y_pred_7 = lm7.predict(X_test_7)

mse7 = mean_squared_error(y_test,y_pred_7)

print("Columns Used: ",len(list(col)))

print("Mean Squared Error: ",mse7,)

print("Mean Squared Error Improvement= ",mse1-mse7)

print("R2_score on test data: ",r2_score(y_test,y_pred_7)," \n")

print("Model Summary\n")

print(lm7.summary())

summary_df=pd.DataFrame({"p":round(lm7.pvalues,3)}).reset_index()

summary_df.head()

# Calculate the VIFs for the new model

vif = pd.DataFrame()

vif['index'] = X_train_6.columns

vif['VIF'] = [variance_inflation_factor(X_train_6.values, i) for i in range(X_train_6.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif.head()

## SINCE ALL P-VALUES ARE NOW BELOW 0.05 WE WILL BE REMOVING COLUMNS WITH HIGH VIF

summary_df=pd.merge(summary_df,vif,on='index')

summary_df=summary_df.sort_values(['p','VIF'],ascending=[0,0])

summary_df.head()
col=list(X_train_6.columns)

col.remove('cylindernumber_three')

X_train_7 = X_train_6[col].copy()

X_train_7 = sm.add_constant(X_train_7)# -> Won't add column 

lm8 = sm.OLS(np.array(y_train),X_train_7).fit()   # Running the linear model

X_test_8 = X_test_7[col].copy()

X_test_8 = sm.add_constant(X_test_8)

y_pred_8 = lm8.predict(X_test_8)

mse8 = mean_squared_error(y_test,y_pred_8)

print("Columns Used: ",len(list(col)))

print("Mean Squared Error: ",mse8)

print("Mean Squared Error Improvement= ",mse1-mse8)

print("R2_score on test data: ",r2_score(y_test,y_pred_8)," \n")

print("Model Summary\n")

print(lm8.summary())

summary_df=pd.DataFrame({"p":round(lm8.pvalues,3)}).reset_index()

summary_df.head()

# Calculate the VIFs for the new model

vif = pd.DataFrame()

vif['index'] = X_train_7.columns

vif['VIF'] = [variance_inflation_factor(X_train_7.values, i) for i in range(X_train_7.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif.head()

## SINCE ALL P-VALUES ARE NOW BELOW 0.05 WE WILL BE REMOVING COLUMNS WITH HIGH VIF

summary_df=pd.merge(summary_df,vif,on='index')

summary_df=summary_df.sort_values(['p','VIF'],ascending=[0,0])

summary_df.head()
col=list(summary_df['index'])

col.remove('const')

# colm=['enginelocation','peakrpm','enginetype_ohcv','wheelbase','enginesize','price']

sns.heatmap(df[col].corr(),annot=True)

#As we can see in the heatmap below most of the variables are weakly correlated.

#Which means we have truly independent variables
'''

BELOW WE CAN SEE THE FINAL MODEL WITH ALL VARIABLES HAVING A P-VALUE LESS THAN 0.05 AND VIF 

VALUE BELOW 5. THUS WE CAN INFER FROM THIS THAT VARIABLES ARE FAIRLY INDEPENDENT AND RELEVANT 

FOR UNDERSTANDING THE DEPENDENT VARIABLE. 

POINTS TO INFER FROM THE MODEL:

    1. THE MEAN SQUARE ERROR IMPROVEMENT i.e 

       MEAN SQUARE ERROR FOR INTIAL MODEL-MEAN SQUARE FOR FINAL MODEL>0

       IMPLIES MEAN SQUARE ERROR HAS IMPROVED FOR THE FINAL MODEL

    2. DIFFERENCE BETWEEN R-SQUARED AND ADJUSTED R-SQUARED IS JUST 0.09, THUS MOST OF THE 

       COLUMNS USED ARE RELEVANT.

    3. THE COEFFICIENT FOR COMPANY NAMES SUCH AS BMW,BUICK ARE POSITIVE WHICH IMPLIES IF THE        

       BRAND NAME THEN THE COEFFICIENT GETS ADDED TO TOTAL PRICE. EG:- IF THE CAR IS A BMW. 

       THEN JUST BY THE VIRTUE OF IT BEING A BMW IT WILL ADD $'8209.05' TO THE ENTIRE VALUE.

       SIMILAR OBSERVATION CAN BE MADE REGARDING THE CAR MANUFACTURERS LIKE TOYOTA WHO              

       MANUFACTURE 'BUDGET' CARS WITH HIGHEST NUMBER OF SALES HAVE A NEGATIVE COEFFICIENT.

'''
# Columns Used:  20

# Mean Squared Error:  4695359.3462523455

# Mean Squared Error Improvement=  187970.6992773842

# R2_score on test data:  0.8952486353281504  



# Model Summary



#                             OLS Regression Results                            

# ==============================================================================

# Dep. Variable:                      y   R-squared:                       0.945

# Model:                            OLS   Adj. R-squared:                  0.936

# Method:                 Least Squares   F-statistic:                     110.7

# Date:                Mon, 26 Aug 2019   Prob (F-statistic):           7.15e-68

# Time:                        21:06:43   Log-Likelihood:                -1288.5

# No. Observations:                 143   AIC:                             2617.

# Df Residuals:                     123   BIC:                             2676.

# Df Model:                          19                                         

# Covariance Type:            nonrobust                                         

# ===========================================================================================

#                               coef    std err          t      P>|t|      [0.025      0.975]

# -------------------------------------------------------------------------------------------

# const                    1.407e+04    377.366     37.276      0.000    1.33e+04    1.48e+04

# aspiration               1775.1463    571.692      3.105      0.002     643.518    2906.775

# enginelocation           9914.0840   2345.701      4.226      0.000    5270.914    1.46e+04

# carwidth                 2106.0231    369.331      5.702      0.000    1374.956    2837.091

# enginesize               4379.1563    338.238     12.947      0.000    3709.635    5048.678

# carbody_convertible      3251.5557   1605.154      2.026      0.045      74.251    6428.860

# cylindernumber_two       6909.7167   1436.870      4.809      0.000    4065.520    9753.913

# fuelsystem_1bbl         -1920.0356    837.464     -2.293      0.024   -3577.745    -262.327

# Company_Name_bmw         8209.0583    979.096      8.384      0.000    6270.998    1.01e+04

# Company_Name_buick       5036.0746   1139.751      4.419      0.000    2780.007    7292.142

# Company_Name_dodge      -2966.9057   1050.066     -2.825      0.006   -5045.447    -888.365

# Company_Name_mazda      -2395.2933    780.216     -3.070      0.003   -3939.683    -850.903

# Company_Name_mitsubishi -4060.0100    856.699     -4.739      0.000   -5755.793   -2364.227

# Company_Name_nissan     -2955.4761    721.615     -4.096      0.000   -4383.870   -1527.082

# Company_Name_peugeot    -2047.2585    886.675     -2.309      0.023   -3802.378    -292.139

# Company_Name_plymouth   -2666.2129    977.065     -2.729      0.007   -4600.254    -732.172

# Company_Name_porsche     4102.8628   1630.279      2.517      0.013     875.826    7329.900

# Company_Name_renault    -5151.7773   1557.709     -3.307      0.001   -8235.167   -2068.388

# Company_Name_toyota     -2300.5547    634.743     -3.624      0.000   -3556.989   -1044.121

# Company_Name_volkswagen -1945.3101    898.556     -2.165      0.032   -3723.947    -166.673

# ==============================================================================

# Omnibus:                       37.717   Durbin-Watson:                   2.085

# Prob(Omnibus):                  0.000   Jarque-Bera (JB):              101.453

# Skew:                           1.022   Prob(JB):                     9.33e-23

# Kurtosis:                       6.584   Cond. No.                         20.0

# ==============================================================================
# Error terms : Since error is randomly distributed we can say the model is appropriate

c = [i for i in range(0,len(y_test.price),1)]

fig = plt.figure()

plt.plot(c,y_test.price-np.array(y_pred_8), color="red", linewidth=2.5, linestyle="-")

fig.suptitle('Error Terms', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                      # X-label

plt.ylabel('ytest-ypred', fontsize=16)                # Y-label
# Actual and Predicted

fig = plt.figure() 

plt.plot(list(range(0,len(y_test.price))),y_test.price, color="green", linewidth=2.5, linestyle="-") #Plotting Actual

plt.plot(list(range(0,len(y_test.price))),np.array(y_pred_8), color="red",  linewidth=2.5, linestyle="-") #Plotting predicted

fig.suptitle('Actual and Predicted', fontsize=20)              

plt.xlabel('Index', fontsize=18)                               

plt.ylabel('Car Price', fontsize=16)                    
fig = plt.figure()

sns.distplot(y_test.price-np.array(y_pred_8))

fig.suptitle('Error Terms', fontsize=25)         

plt.xlabel('y_test-y_pred', fontsize=22)              

plt.ylabel('Index', fontsize=16)                       
# Now let's check the Root Mean Square Error of our model.

print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred_8)))

print('R2_SCORE:', r2_score(y_test, y_pred_8))