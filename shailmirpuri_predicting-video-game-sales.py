# Load Libaries Needed

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 
# Read the data

data_path='/kaggle/input/videogamesales/vgsales.csv'

df=pd.read_csv(data_path)

# Print the first 5 rows

df.head()
# Look at all the variables in the columns

df.columns
#Explore the data type of each column

df.info()
# Get a snapshot of the data's central tendencies 

df.describe()
# Let's see how the Global Sales are distritubed for all the data 

f,ax=plt.subplots(figsize=(10,5))

sns.distplot(df['Global_Sales'])

plt.title("Distribution of Global Sales for Video Games")

# Calculate Skewness and Kurtosis 

print ("Skewness: " + str(df['Global_Sales'].skew()))

print ("Kurtosis: "+str(df['Global_Sales'].kurtosis()))
#Numerical Variables 

num= ['Year','NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']

df_num=df[num]

for var in df_num:

    if var!='Global_Sales':

        f,ax=plt.subplots(figsize=(5,5))

        sns.distplot(df_num[var])

#Calculate Skewness and Kurtosis for each numerical variable

for var in df_num: 

    if var != "Global_Sales":

        print ('Skewness for '+ var + ' is: '+ str(df[var].skew()))

        print ('Kurtosis for '+ var + ' is: '+ str(df[var].kurtosis()))
#Calculate Correlation between numeric variables

sns.heatmap(df_num.corr(),annot=True,cmap="YlGnBu")
#Categorical Variables

cat=['Platform','Genre', 'Publisher']

df_cat=df[cat]

for var in df_cat:

    f,ax=plt.subplots(figsize=(16,8))

    sns.barplot(df_cat[var].value_counts().index,df_cat[var].value_counts()).set_title('Distrubution of ' + var)

    plt.xlabel(var)

    plt.ylabel('Frequency')

    plt.show()
# Sales across different platforms:

f,ax=plt.subplots(figsize=(18,6))

sns.boxplot(df['Platform'],df['Global_Sales'])

plt.show()



#Calculate average sale price for each platform:

df2=df.groupby(['Platform'])['Platform','Global_Sales'].mean().sort_values(by='Global_Sales', ascending=False).reset_index()

df2
# Sales over the years 

df3 =df.groupby(['Year'])['Global_Sales'].mean().sort_values(ascending=False).reset_index()

print(df3)



# Scatter Plot

f,ax=plt.subplots(figsize=(18,6))

plt.scatter(df['Year'],df['Global_Sales'])

# Best Selling Genres:

df4 =df.groupby(['Genre'])['Genre','Global_Sales'].mean().sort_values(by='Global_Sales', ascending=False).reset_index()

print(df4)



# Boxplot comparison between genres

f,ax=plt.subplots(figsize=(18,6))

sns.boxplot(df['Genre'],df['Global_Sales'])

plt.ylim([0,10])
# Most Successful Publishers:

df5 =df.groupby(['Publisher'])['Publisher','Global_Sales'].mean().sort_values(by='Global_Sales', ascending=False).reset_index()

print(df5.head(10))



# Boxplot comparing Publishers:

f,ax=plt.subplots(figsize=(18,6))

sns.boxplot(df['Publisher'],df['Global_Sales'])
# Comparing Sales of Major Game Series

series=['Fifa','Mario','Call of Duty','Grand Theft Auto', 'Pokemon','Halo','Wii','NBA']



for ser in series:

    M = df[df['Name'].str.contains(ser, regex=False, case=False, na=False)].copy()

    f,ax=plt.subplots(figsize=(16,5))

    plt.scatter(M['Year'],M['Global_Sales'])

    plt.title(ser+' Sales Over Time')

    plt.xlabel('Year')

    plt.ylabel('Global Sales')

    plt.show()



# Compare average sales for each major game series

series=['Fifa','Mario','Call of Duty','Grand Theft Auto', 'Pokemon','Halo','Wii','NBA']

for ser in series:

    game_series={}

    M = df[df['Name'].str.contains(ser, regex=False, case=False, na=False)].copy()

    average= round(M['Global_Sales'].median(),2)

    correlation=M['Global_Sales'].corr(M['Year'])

    rounded_corr=round(correlation,2)

    print('The Median For {} Series is {}, its Correlation with Years is {}'.format(ser,average,rounded_corr))
# Compare Type of Device Game Sales 

print(df['Platform'].unique())

# 3 types: Portable, PC and Console

PC=['PC']

Portable = ['GB','DS','GBA','3DS','PSP','PSV','WS','GG']

df_portable=df[(df.Platform =='GB')|(df.Platform =='DS')|(df.Platform =='GBA')|(df.Platform =='3DS')|(df.Platform =='PSP')|(df.Platform =='PSV')

              |(df.Platform =='WS')|(df.Platform =='GG')]

df_PC=df[(df.Platform=='PC')]



df_console= df[(df.Platform =='Wii')|(df.Platform =='NES')|(df.Platform =='X360')|(df.Platform =='PS3')|(df.Platform =='PS2')|(df.Platform =='SNES')

              |(df.Platform =='PS4')|(df.Platform =='N64')|(df.Platform =='PS')|(df.Platform =='XB')|(df.Platform =='2600')|(df.Platform =='XOne')

             |(df.Platform =='GC')|(df.Platform =='WiiU')|(df.Platform =='GEN')|(df.Platform =='DC')

            |(df.Platform =='SAT')|(df.Platform =='SCD')|(df.Platform =='NG')|(df.Platform =='TG16')|(df.Platform =='3DO')

            |(df.Platform =='PCFX')]

d={'Portable':df_portable,'PC': df_PC,'Console':df_console}

for name,n in d.items():

    f,ax=plt.subplots(figsize=(10,6))

    plt.scatter(n['Year'],n['Global_Sales'])

    plt.title(name + ' Device Sales Over Time')

    plt.xlabel('Year')

    plt.ylabel('Global Sales')

    plt.show()

    average= n['Global_Sales'].median()

    print ("The Median Sales for {} devices is {}".format(name,average))





len(df['Platform'])
#Console, Portable or PC

Device=[]

for i in range(0,16598):

    if df['Platform'][i] in Portable:

        Device.append('Portable')

    elif df['Platform'][i] in PC:

        Device.append('PC')

    else:

        Device.append('Console')

df['Device']=Device

df.head()

        
# Adding Feature of Game Series

series=['FIFA','Mario','Call of Duty','Grand Theft Auto', 'Pokemon','Halo','Wii','NBA']

for ser in series:

    df[ser]=df['Name'].str.contains(ser, regex=False, case=False, na=False)

df.head(10)
# Percentage of Total Sales Per Region

regions_sales=['NA_Sales','EU_Sales','JP_Sales','Other_Sales']

for reg in regions_sales:

    A=df[reg]

    B=df['Global_Sales']

    df[reg+' As a percentage of Total']=round(100*(A/B),2)

df
# Splitting Data into input and output variables

features= ['Year', 'Genre', 'Publisher','Device',

       'FIFA', 'Mario', 'Call of Duty', 'Grand Theft Auto', 'Pokemon', 'Halo',

       'Wii', 'NBA', 'NA_Sales As a percentage of Total',

       'EU_Sales As a percentage of Total',

       'JP_Sales As a percentage of Total',

       'Other_Sales As a percentage of Total']

X=df[features]

X

y=df['Global_Sales']
# Checking for Missing values

print(X.isnull().sum())
# Investigating which rows have missing values

df[df.isnull().any(axis=1)]

# For those with Publishers as NaN we will fill them with unknown

X.Publisher.fillna('Unknown',inplace=True)

X.Publisher.isnull().sum()
# Dealing with missing year value:

df[df.Year.isnull()]

# From look at this dataset there does not seem to be a pattern for missing years. We will therefore fill it by its median since 

# this will be a better measure of center than mean due to the left skewed distrbution of Years in this dataset

f=df.Year.median()

X.Year.fillna(f,inplace=True)

# We should indicate in which records we have imputed years column with the median

X['Year was Missing']=df.Year.isnull()

X[X['Year was Missing']==True]

# Confirm all missing values are filled

X.isnull().sum()
# Now let's move on to the categorical variables

# For Series Type we have already encoded it using OneHotEncoding. The only other two variables left are Genre, Publisher and Device Type

# Check the cardinality for each of these variables

for col in ['Genre','Publisher','Device']:

    print('The cardinality of {} is {}'.format(col,X[col].nunique()))
# For Device Type we can use One-Hot Encoding 

from sklearn.preprocessing import OneHotEncoder

OH_Encoder=OneHotEncoder(sparse=False,handle_unknown='ignore')

low_card_col=['Device']

OH_cols_X=pd.DataFrame(OH_Encoder.fit_transform(X[low_card_col]))

OH_cols_X.index=X.index

other=X.drop(low_card_col,axis=1)

X_processed=pd.concat([other,OH_cols_X], axis=1)
# For the high cardinality columns we shall use label encoding to minimise the data transformations needed to take place

high_cardinality =['Genre','Publisher']

label_X=X_processed.copy()

from sklearn.preprocessing import LabelEncoder

label_encoder= LabelEncoder()

for col in high_cardinality:

    label_X[col]=label_encoder.fit_transform(X[col])
# Winsorization

from scipy.stats.mstats import winsorize

y_w=winsorize(y,limits=0.0001)

y_w
# Linear Regression

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

lin=LinearRegression()

cv=-1*cross_val_score(lin,label_X,y_w,cv=5,scoring='neg_mean_absolute_error')

print(cv)

print(cv.mean())
# Decision Tree Regressor

from sklearn import tree

dt=tree.DecisionTreeRegressor(max_leaf_nodes=950,random_state=1)

cv=-1*cross_val_score(dt,label_X,y_w,cv=5,scoring='neg_mean_absolute_error')

print(cv)

print(cv.mean())
# Random Forest Regressor

from sklearn import ensemble

rf=ensemble.RandomForestRegressor(n_estimators=100,random_state=1)

cv=-1*cross_val_score(rf,label_X,y_w,cv=5,scoring='neg_mean_absolute_error')

print(cv)

print(cv.mean())
# Bagging  Regressor

bag=ensemble.BaggingRegressor(random_state=1)

cv=-1*cross_val_score(bag,label_X,y_w,cv=5,scoring='neg_mean_absolute_error')

print(cv)

print(cv.mean())
# Extra Trees Regressor

etr=ensemble.ExtraTreesRegressor(random_state=1,n_estimators=100)

cv=-1*cross_val_score(etr,label_X,y_w,cv=5,scoring='neg_mean_absolute_error')

print(cv)

print(cv.mean())
# Gradient Boosting Regressor

gb=ensemble.GradientBoostingRegressor(random_state=1)

cv=-1*cross_val_score(gb,label_X,y_w,cv=5,scoring='neg_mean_absolute_error')

print(cv)

print(cv.mean())
#Now let's try combining all the models with a average MAE of lower than 0.57 in a Voting Regressor:

from sklearn.ensemble import VotingRegressor

voting_reg=VotingRegressor(estimators=[('dt',dt),('rf',rf),('gb',gb),('bag',bag),('etr',etr)])

cv=-1*cross_val_score(voting_reg,label_X,y_w,cv=5,scoring='neg_mean_absolute_error')

print(cv)

print(cv.mean())
# Model Tuning

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



def model_performance(model,name):

    print(name)

    print('Best Score: {}'.format(model.best_score_))

    print('Best Parameters: {}'.format(model.best_params_))
# Random Forest Random Search

rf

param={

    'n_estimators':[100,500,1000]}

reg_rf_rs=GridSearchCV(rf, param_grid=param, cv = 5, verbose = True, n_jobs = -1)

best_reg_rf_rnd = reg_rf_rs.fit(label_X,y_w)

model_performance(best_reg_rf_rnd,'Random Forest')
# Most Important Features for Random Forest

best_rf = best_reg_rf_rnd.best_estimator_.fit(label_X,y_w)

feat_importances = pd.Series(best_rf.feature_importances_, index=label_X.columns)

feat_importances.plot(kind='barh')
# Best RF Model

best_rf=best_reg_rf_rnd.best_estimator_



#Developing Voting Model

voting_reg_1 = VotingRegressor(estimators=[('dt',dt),('rf',best_rf),('gb',gb),('bag',bag),('etr',etr)])

cv=-1*cross_val_score(voting_reg_1,label_X,y_w,cv=5,scoring='neg_mean_absolute_error')

print('Cross Val Score: {}'.format(cv))

print('Average Cross Val Score: {}'.format(cv.mean()))
# Try and improve weights for the best voting model:

params = {'weights' : [[1,1,1,1,1],[1,2,1,1,1],[1,1,2,1,1],[1,2,2,1,1]]}



vote_weight = GridSearchCV(voting_reg_1, param_grid = params, cv = 5, verbose = True, n_jobs = -1)

best_reg_weight = vote_weight.fit(label_X,y_w)

model_performance(best_reg_weight,'VR Weights')



best_vote=best_reg_weight.best_estimator_
# Split Data into Train and Test

from sklearn.model_selection import train_test_split

train_X,test_X,train_y,test_y=train_test_split(label_X,y_w,random_state=1,test_size=1000)



# Voting Regressor

# Train Model and Make predictions

best_vote.fit(train_X,train_y)

pred=best_vote.predict(test_X)



# Compute MAE and R2

from sklearn import metrics

mae=metrics.mean_absolute_error(test_y,pred)

r2=metrics.r2_score(test_y,pred)

print('The Mean Absolute Error for our final model is {}'.format(mae))

print('The R-squared score for our final model is {}'.format(r2))
feature_name=(test_X.columns.tolist())

feature_name[-3]='Console'

feature_name[-2]='PC'

feature_name[-1]='Portable'
# Explore feature importance

import eli5

from eli5.sklearn import PermutationImportance

perm = PermutationImportance(best_vote,random_state=1).fit(test_X, test_y)

eli5.show_weights(perm,feature_names = feature_name)