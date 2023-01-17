# suppres warnings

import warnings

warnings.filterwarnings('ignore')
# Import required libraries

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import r2_score, mean_squared_error



import statsmodels.api as sm
# reading data set

df = pd.read_csv('../input/boom-bike-dataset/bike_sharing_data.csv')

df.head()
# getting insights of dataframe

df.shape
# getting descriptive insights of dataframe

df.info()
# Check for any duplicate entries

df.duplicated().sum()
# drop columns instance, dteday, casual, registered and atemp

df.drop(['instant', 'dteday','casual','registered','atemp'], axis=1, inplace=True)
# After droppping the variables checking the columns abnd rows in the dataframe

df.shape
# identify categorical variables

cat_vars = ['season','yr','mnth','holiday','weekday', 'workingday','weathersit']



# identify numeric variables

num_vars = ['temp', 'hum','windspeed','cnt']
# convert dtype of categorical variables

df[cat_vars] = df[cat_vars].astype('category')
# get insights of numeric variable

df.describe()
# get the insights of categorical variables

df.describe(include=['category'])
# maped the season column according to descripttions

df['season'] = df['season'].map({1:'spring', 2:'summer', 3:'fall', 4:'winter'})



# maped the weekday column according to descriptin

df['weekday'] = df['weekday'].map({0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'})





# maped mnth column values (1 to 12 ) as (jan to dec) respectively

df['mnth'] = df['mnth'].map({1:'jan', 2:'feb', 3:'mar', 4:'apr', 5: 'may', 6: 'jun', 7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct',

                             11: 'nov', 12:'dec'})



#  maped weathersit column

df['weathersit'] = df['weathersit'].map({1: 'Clear_FewClouds', 2: 'Mist_Cloudy', 3: 'LightSnow_LightRain', 4: 'HeavyRain_IcePallets'})
# Check the data info before proceeding for analysis

df.info()
# visualise the pattern of demand (target variable - 'cnt') over period of two years

plt.figure(figsize=(20,5))

plt.plot(df.cnt)

plt.show()
# Visualising numerical varibles



# selecting numerical variables

var = df.select_dtypes(exclude = 'category').columns



# Box plot

col = 2

row = len(var)//col+1



plt.figure(figsize=(12,8))

plt.rc('font', size=12)

for i in list(enumerate(var)):

    plt.subplot(row, col, i[0]+1)

    sns.boxplot(df[i[1]])    

plt.tight_layout()   

plt.show()
# get percentage outlier for hum and windspeed



# function to get outlier percentage

def percentage_outlier(x):

    iqr = df[x].quantile(0.75)-df[x].quantile(0.25)

    HL = df[x].quantile(0.75)+iqr*1.5

    LL = df[x].quantile(0.25)-iqr*1.5

    per_outlier = ((df[x]<LL).sum()+(df[x]>HL).sum())/len(df[x])*100

    per_outlier = round(per_outlier,2)

    return(per_outlier)



print('Percentage of outlier (hum): ', percentage_outlier('hum'))

print('Percentage of outlier (windspeed): ', percentage_outlier('windspeed'))
# # # Visulalising Categorical Variables using pie chart



df_piplot=df.select_dtypes(include='category')

plt.figure(figsize=(18,16))

plt.suptitle('pie distribution of categorical features', fontsize=20)

for i in range(1,df_piplot.shape[1]+1):

    plt.subplot(3,3,i)

    f=plt.gca()

    f.set_title(df_piplot.columns.values[i-1])

    values=df_piplot.iloc[:,i-1].value_counts(normalize=True).values

    index=df_piplot.iloc[:,i-1].value_counts(normalize=True).index

    plt.pie(values,labels=index,autopct='%1.0f%%')

# plt.tight_layout(pad = 0.5)

plt.show()
# # Visulalising Categorical Variables

# # selecting categorical variables

# var = df.select_dtypes(include='category').columns



# # Box plot

# col = 3

# row = len(var)//col+1



# plt.figure(figsize=(12,12))

# # plt.rc('font', size=12)

# for i in list(enumerate(var)):

#     plt.subplot(row, col, i[0]+1)

#     sns.countplot(df[i[1]])

#     plt.xticks(rotation = 90)

# plt.tight_layout(pad = 1.0)

# plt.show()
# pairplot for continuous data type

sns.pairplot(df.select_dtypes(['int64','float64']), diag_kind='kde')

plt.show()
# look at the correaltion between continous varibales using heat map

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.show()
# Box plot for categorical variables

col = 3

row = len(cat_vars)//col+1



plt.figure(figsize=(15,12))

for i in list(enumerate(cat_vars)):

    plt.subplot(row,col,i[0]+1)

    sns.boxplot(x = i[1], y = 'cnt', data = df)

    plt.xticks(rotation = 90)

plt.tight_layout(pad = 1)    

plt.show()
# get dummy variables for season, weekday, mnth and weathersit

dummy_vars = pd.get_dummies(df[['season','weekday','mnth','weathersit']],drop_first=True)



# concat the dummy df with original df

df = pd.concat([df,dummy_vars], axis = 1)



# drop season column

df.drop(['season','weekday','mnth','weathersit'], axis=1, inplace=True)



df.head()
# check data frame

df.shape
# Check datafrmae

df.info()
# Convert categorical columns to numeric 

df[['yr','holiday','workingday']]= df[['yr','holiday','workingday']].astype('uint8')

df.info()
# Split train test dataset

df_train, df_test = train_test_split(df, train_size = 0.7, random_state = 10 )

print(df_train.shape)

print(df_test.shape)
# Scaling of train set



# instantiate an object

scaler = MinMaxScaler()



# fit and transform on training data

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

df_train.head()
# check test dataset before scaling

df_test.head()
# transform test dataset 

df_test[num_vars] = scaler.transform(df_test[num_vars])

df_test.head()
# Creating X and y data dataframe for train set

y_train = df_train.pop('cnt')

X_train = df_train

X_train.head()
# Creating X and y data dataframe for test set

y_test = df_test.pop('cnt')

X_test = df_test



X_test.head()
# Checking variables for for X_train columns

X_train.columns
# Running RFE to select 15 number of varibles

# Create object

lm = LinearRegression()

# fit model

lm.fit(X_train, y_train)

# run RFE

rfe = RFE(lm, 15)

rfe = rfe.fit(X_train, y_train)



# Select columns

col = X_train.columns[rfe.support_]

col
# Creating X_train_rfe with RFE selected variables

X_train_rfe = X_train[col]
# create function for stats linear model 

def sm_linearmodel(X_train_sm):

    #Add constant

    X_train_sm = sm.add_constant(X_train_sm)



    # create a fitted model (1st model)

    lm = sm.OLS(y_train,X_train_sm).fit()

    return lm
# Function to calculate VIF

# calculate VIF

def vif_calc(X):

    vif = pd.DataFrame()

    vif['Features'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif['VIF'] = round(vif['VIF'],2)

    vif = vif.sort_values(by='VIF', ascending = False)

    return vif
# Create 1st stats model and look for summary and VIF

lm_1 = sm_linearmodel(X_train_rfe)

print(lm_1.summary())



# Calculate VIF

print(vif_calc(X_train_rfe))
# Loop to remove P value variables >0.05 in bstep mannen and update model



pvalue = lm_1.pvalues

while(max(pvalue)>0.05):

    maxp_var = pvalue[pvalue == pvalue.max()].index

    print('Removed variable:' , maxp_var[0], '    P value: ', round(max(pvalue),3))

    

    # drop variable with high p value

    X_train_rfe = X_train_rfe.drop(maxp_var, axis = 1)

    lm_1 = sm_linearmodel(X_train_rfe)

    pvalue = lm_1.pvalues

    

    
# Look for sumamry of model

print(lm_1.summary())



# Calculate VIF

print(vif_calc(X_train_rfe))
# drop varible having high VIF

X_train_new = X_train_rfe.drop(['hum'],axis = 1)



# Create stats model and look for summary

lm_2 = sm_linearmodel(X_train_new)

print(lm_2.summary())



# Calculate VIF

print(vif_calc(X_train_new))
# drop varible having high VIF

X_train_new = X_train_new.drop(['season_fall'],axis = 1)



# Create stats model and look for summary

lm_3 = sm_linearmodel(X_train_new)

print(lm_3.summary())



# Calculate VIF

print(vif_calc(X_train_new))
# drop varible having high VIF

X_train_new = X_train_new.drop(['mnth_mar'],axis = 1)



# Create stats model and look for summary

lm_4 = sm_linearmodel(X_train_new)

print(lm_4.summary())



# Calculate VIF

print(vif_calc(X_train_new))
# drop varible having high VIF

X_train_new = X_train_new.drop(['mnth_oct'],axis = 1)



# Create stats model and look for summary

lm_5 = sm_linearmodel(X_train_new)

print(lm_5.summary())



# Calculate VIF

print(vif_calc(X_train_new))
# List down final model varibales and its coefficients



# assign final model to lm_final

lm_final = lm_5



# list down and check variables of final model

var_final = list(lm_final.params.index)

var_final.remove('const')

print('Final Selected Variables:', var_final)



# Print the coefficents of final varible

print('\033[1m{:10s}\033[0m'.format('\nCoefficent for the variables are:'))

print(round(lm_final.params,3))
# Select final variables from the test dataset

X_train_res = X_train[var_final]
#Add constant

X_train_res = sm.add_constant(X_train_res)



# predict train set

y_train_pred = lm_final.predict(X_train_res)
# distrubition plot for residue

res = y_train - y_train_pred

sns.distplot(res)

plt.title('Error terms')

plt.show()
# Error terms train set

c = [i for i in range(1,len(y_train)+1,1)]

fig = plt.figure(figsize=(8,5))

plt.scatter(y_train,res)

fig.suptitle('Error Terms', fontsize=16)              # Plot heading 

plt.xlabel('Y_train_pred', fontsize=14)                      # X-label

plt.ylabel('Residual', fontsize=14)   
# check dataframe for the test set

df_test.head()
# select final variables from X_test

X_test_sm = X_test[var_final]

X_test_sm.head()
# add constant

X_test_sm = sm.add_constant(X_test_sm)

X_test_sm.head()
# predict test dataset

y_test_pred = lm_final.predict(X_test_sm)
# Get R-Squared fro test dataset

r2_test = r2_score(y_true = y_test, y_pred = y_test_pred)

print('R-Squared for Test dataset: ', round(r2_test,3))
# Adj. R-Squared for test dataset

N= len(X_test)          # sample size

p =len(var_final)     # Number of independent variable

r2_test_adj = round((1-((1-r2_test)*(N-1)/(N-p-1))),3)

print('Adj. R-Squared for Test dataset: ', round(r2_test_adj,3))
# Mean Sqare Error

mse = mean_squared_error(y_test, y_test_pred)

print('Mean_Squared_Error :' ,round(mse,4))
res_test = y_test - y_test_pred

plt.title('Error Terms', fontsize=16) 

sns.distplot(res_test)

plt.show()
# Error terms

c = [i for i in range(1,len(y_test)+1,1)]

fig = plt.figure(figsize=(8,5))

plt.scatter(y_test,res_test)

fig.suptitle('Error Terms', fontsize=16)              # Plot heading 

plt.xlabel('Y_test_pred', fontsize=14)                      # X-label

plt.ylabel('Residual', fontsize=14)   
# Plotting y_test and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y_test, y_test_pred)

fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y_test', fontsize = 18)                          # X-label

plt.ylabel('y_test_pred', fontsize = 16)      
# Print R Squared and adj. R Squared

print('R- Sqaured train: ', round(lm_final.rsquared,2), '  Adj. R-Squared train:', round(lm_final.rsquared_adj,3) )

print('R- Sqaured test : ', round(r2_test,2), '  Adj. R-Squared test :', round(r2_test_adj,3))



# Print the coefficents of final varible

print('\033[1m{:10s}\033[0m'.format('\nCoefficent for the variables are:'))

print(round(lm_final.params,3))