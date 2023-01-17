import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) 

carFilePath = '../input/CarPrice_Assignment.csv'
carsDF = pd.read_csv(carFilePath,encoding = "ISO-8859-1", low_memory=False)
carsDF.info()
carsDF.head()
# symboling: -2 (least risky) to +3 most risky
carsDF['symboling'].astype('category').value_counts()
# aspiration: property of a combustible engine
carsDF['aspiration'].astype('category').value_counts()
# drivewheel: frontwheel, rear wheel or four-wheel drive 
carsDF['drivewheel'].astype('category').value_counts()
# wheelbase: distance between centre of front and rear wheels
sns.distplot(carsDF['wheelbase'])
plt.show()
# curbweight: weight of car without occupants or baggage
sns.distplot(carsDF['curbweight'])
# stroke: volume of the engine (the distance traveled by the 
# piston in each cycle)
sns.distplot(carsDF['stroke'])
plt.show()
# compression ration: ration of volume of compression chamber 
# at largest capacity to least capacity
sns.distplot(carsDF['compressionratio'])
plt.show()
# target variable: price of car

fig, ax= plt.subplots(1,2,figsize=(15,5))

sns.distplot(carsDF['price'], norm_hist=False, kde=False, ax=ax[0], color='red')
ax[0].set_xlabel('Price of Car')
ax[0].set_ylabel('Count of Cars',size=12)
ax[0].set_title('Count of Cars By Price',size=15,weight="bold")

sns.distplot(carsDF['price'], kde=True, ax=ax[1], color='blue')
ax[1].set_xlabel('Price of Car')
ax[1].set_ylabel('Relative Frequency of Cars',size=12)
ax[1].set_title('Density of Cars By Price',size=15,weight="bold")
#creating df  with numeric var's only whose data type is either float64 or int64
numeric_vars_df=carsDF.select_dtypes(include=['float64','int64'])
numeric_vars_df.head()
# we do not need symboling (although it gives risk type but it actually is a categorical variable) and car_ID.
# Let's drop them from numeric variables.
numeric_vars_df = numeric_vars_df.drop(['symboling', 'car_ID'], axis=1)
numeric_vars_df.head()
# Let's do a pairplot to see if we can derive something out of it.
plt.figure(figsize=(20, 10))
sns.pairplot(numeric_vars_df)
# we will plot all the numeric variables one by one through a loop

for feature, column in enumerate (numeric_vars_df.columns):
    plt.figure(feature)
    sns.scatterplot(x=numeric_vars_df[column],y=numeric_vars_df['price'])
corr=numeric_vars_df.corr()

plt.figure(figsize=(15,8))
sns.heatmap(corr,annot=True,cmap="YlGnBu")
carsDF.info()
# converting symboling to categorical
carsDF['symboling'] = carsDF['symboling'].astype('object')
carsDF.info()
car_names_after_split = carsDF['CarName'].apply(lambda x: x.split(" ")[0])
car_names_after_split[:10]
carsDF['company']=car_names_after_split
carsDF['company'].value_counts()

# many car names have bogus values like toyouta, vk and porcshce etc.
# we need to fix incorrect spelling and get carnames column in order
#toyota
carsDF.loc[(carsDF['company']=="toyouta"),"company"]="toyota"

# nissan
carsDF.loc[carsDF['company'] == "Nissan", 'company'] = 'nissan'

# mazda
carsDF.loc[carsDF['company'] == "maxda", 'company'] = 'mazda'

#volkswagen
carsDF.loc[(carsDF['company']=="vw")|(carsDF['company']=="vokswagen"),"company"]="volkswagen"

#porsche
carsDF.loc[(carsDF['company']=="porcshce"),"company"]="porsche"

carsDF['company'].value_counts()
# drop carname variable since we do not need it anymore
carsDF = carsDF.drop('CarName', axis=1)
# create X & y subsets from carsDF for easy analysis
X=carsDF.drop(columns=['price',"car_ID"])
y=carsDF['price']
# creating categorical variables, basically, these are the variables who data type is object.
categorical_cars_vars = X.select_dtypes(include=['object'])
categorical_cars_vars.head()
# creating dummy variables from the list of categorical variables just created.
dummy_vars_df = pd.get_dummies(categorical_cars_vars, drop_first=True)
dummy_vars_df.head()
# since we have the dummy variables now, get can drop the categorical variables from X.
X=X.drop(columns=categorical_cars_vars)
X.head()
# add the dummy variables to X and store in a separate df.
X_merged=pd.merge(X,dummy_vars_df,on=X.index)
X_merged.head()
# Let's have a look at the columns of merged df.
X_merged.columns
# get rid of the 'key_0' column after the merge
X_merged.drop(columns='key_0',inplace=True)
X_merged.info()
from sklearn.preprocessing import scale

cols=X_merged.columns
X_scaled=pd.DataFrame(scale(X_merged))
X_scaled.columns=cols
X_scaled.columns
X_scaled.describe()
# split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)
# import necessary libraries and start modelling, start with complete dataset
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

lm=LinearRegression()
lm.fit(X_train,y_train)

y_pred_test=lm.predict(X_test)
y_pred_train=lm.predict(X_train)
# next step is to calculate the r2 scores, RMSE

from sklearn.metrics import r2_score

print('R-squared on train data: {}'.format(r2_score(y_true=y_train, y_pred=y_pred_train)))
print('R-squared on test data: {}'.format(r2_score(y_true=y_test, y_pred=y_pred_test)))

#Standard error/RMSE
error_train=y_pred_train-y_train
error_test=y_pred_test-y_test

print('RMSE on train data: {}'.format(((error_train**2).mean())**0.5))
print('RMSE on test data: {}'.format(((error_test**2).mean())**0.5))
from sklearn.feature_selection import RFE
import statsmodels.api as sm

n_features_list = list(range(4, 31)) #checking for optimal number of features between 4 to 30
train_adjusted_r2 = []
train_r2 = []
test_r2 = []
train_RMSE=[]
test_RMSE=[]

for n_features in range(4, 31):

    # RFE with n features
    lm = LinearRegression()

    # specifying number of features
    rfe_n = RFE(estimator=lm, n_features_to_select=n_features)

    # fit with n features
    rfe_n.fit(X_train, y_train)

    # selecting features selected by rfe_n
    col_n = X_train.columns[rfe_n.support_] #rfe_n.support_: returns an array with boolean values to indicate whether 
    #an attribute was selected using RFE

    # training & test data for n selected columns
    X_train_rfe_n = X_train[col_n]
    X_test_rfe_n = X_test[col_n]


    # add a constant to the model
    X_train_rfe_n = sm.add_constant(X_train_rfe_n)


    X_test_rfe_n = sm.add_constant(X_test_rfe_n, has_constant='add')

    
    
    # fitting the model with n featues
    lm_n = sm.OLS(y_train, X_train_rfe_n).fit()
    
    
    # # Making predictions
    y_pred_test = lm_n.predict(X_test_rfe_n)
    y_pred_train = lm_n.predict(X_train_rfe_n)
    
    
    #Calculating evaluation metrics
    
    #R-square
    train_adjusted_r2.append(lm_n.rsquared_adj)
    train_r2.append(lm_n.rsquared)
    test_r2.append(r2_score(y_test, y_pred_test))
    
    #RMSE/stan. error
    error_test=y_pred_test-y_test
    error_train=y_pred_train-y_train
    
    test_RMSE.append(((error_test**2).mean())**0.5)
    train_RMSE.append(((error_train**2).mean())**0.5)
# plotting r2 and RMSE against n_features
fig,ax=plt.subplots(2,1,figsize=(13, 9))
ax[0].plot(n_features_list, train_r2,'b', label="r2_train data")
ax[0].plot(n_features_list, test_r2,'g', label="r2_test data")
ax[0].set_xlabel('Count of Features')
ax[1].plot(n_features_list, train_RMSE, 'b',label="RMSE_train data")
ax[1].plot(n_features_list, test_RMSE, 'g',label="RMSE_test data")
ax[1].set_xlabel('Count of Features')

ax[0].legend(loc='upper left')
ax[1].legend(loc='upper left')
plt.show()
lm=LinearRegression()
rfe=RFE(lm,13)
rfe.fit(X_train,y_train)

col=X_train.columns[rfe.support_] #obtaining fetaure names of 13 most imp 

#making new Df's with  13 most imp features as per RFE algorithm
X_train_13= X_train[col]
X_test_13 = X_test[col]


# add a constant to the model
X_train_13 = sm.add_constant(X_train_13,has_constant='add')
X_test_13 = sm.add_constant(X_test_13,has_constant='add')


#fit OLS model
lm_sm=sm.OLS(y_train,X_train_13).fit()


#making predictions
y_pred_train=lm_sm.predict(X_train_13)
y_pred_test=lm_sm.predict(X_test_13)


#evlaution metrics

#R-sqaure
train_r2=lm_sm.rsquared
test_r2=r2_score(y_pred_test, y_test)

#RMSE/stan. error
error_test=y_pred_test-y_test
error_train=y_pred_train-y_train
    
test_RMSE=(((error_test**2).mean())**0.5)
train_RMSE=(((error_train**2).mean())**0.5)

print('----------------------R2 scores for Test and Train--------------------------------')
print("R2 Score for test data is {}".format(test_r2))
print("R2 Score for train data is {}".format(train_r2))


print('----------------------STANDARD ERROR/RMSE results for Test and Train----------------')
print("RMSE for test data is {}".format(test_RMSE))
print("RMSE for train data is {}".format(train_RMSE))

print(lm_sm.summary())

fig, ax=plt.subplots(figsize=(15,6))
sns.lineplot(x=y_test.index,y=y_test,label='Actuals',color='blue',ax=ax)
sns.lineplot(x=y_test.index,y=y_pred_test,label='Predicted',color='red',ax=ax)
ax.set_title('Car Price: Actuals vs Predicted')
ax.set_ylabel('Price')
ax.set_xlabel('Index')
# get the features as given by the rfe technique
rfe_features=lm_sm.params.index
# get rid of the constant from our list of features
rfe_features=rfe_features[1:]
rfe_features
# Let's take a look how our X_train look like with these features
X_train[rfe_features].head()
# We will do this by plotting graphs of Observed Values Vs Predicted Values and Residuals Vs Predicted Values.

def test_linearity_of_model(model, y):
    
    fitted_vals = model.predict()
    residuals = model.resid
    
    sns.set_style('darkgrid')
    fig,ax=plt.subplots(1,2, figsize=(15,4))
    
    sns.regplot(x=fitted_vals, y=y, lowess=True, ax=ax[0], line_kws={'color': 'red'})
    ax[0].set_title('Observed Values vs. Predicted Values', fontsize=16)
    ax[0].set_xlabel('Predicted Values', fontsize=13)
    ax[0].set_ylabel('Observed', fontsize=13)
    
    sns.regplot(x=fitted_vals,y=residuals,lowess=True,ax=ax[1],line_kws={'color': 'red'})
    ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
    ax[1].set_xlabel('Predicted', fontsize=13)
    ax[1].set_ylabel('Residuals', fontsize=13)
    
test_linearity_of_model(lm_sm, y_train)
# This assumption means that the variance around the regression line is the same for all values of the predictor variable (X).
 
import statsmodels.stats.api as sms

def test_homoscedasticity_of_model(model):
    
    fitted_vals = model.predict()
    residuals = model.resid
    resids_standardized = model.get_influence().resid_studentized_internal
    
    sns.set_style('darkgrid')
    
    
    fig, ax = plt.subplots(1,2,figsize=(15,4))

    sns.regplot(x=fitted_vals, y=residuals, lowess=True, ax=ax[0], line_kws={'color': 'red'})
    ax[0].set_title('Residuals vs Fitted', fontsize=16)
    ax[0].set_xlabel('Predicted', fontsize=13)
    ax[0].set_ylabel('Residuals', fontsize=13)

    sns.regplot(x=fitted_vals, y=np.sqrt(np.abs(resids_standardized)), lowess=True, ax=ax[1], line_kws={'color': 'red'})
    ax[1].set_title('Scale-Location', fontsize=16)
    ax[1].set_xlabel('Predicted', fontsize=13)
    ax[1].set_ylabel('Standardized Residuals', fontsize=13)

test_homoscedasticity_of_model(lm_sm)
from scipy import stats

def test_normality_of_residuals(model):
   
    sm.ProbPlot(model.resid).qqplot(line='s');
    plt.title('Q-Q plot');
    

    jb = stats.jarque_bera(model.resid)
    sw = stats.shapiro(model.resid)
    ad = stats.anderson(model.resid, dist='norm')
    ks = stats.kstest(model.resid, 'norm')
    
    print(f'Jarque-Bera test statistics: {jb[0]:.4f}, p-value: {jb[1]}')
    print(f'Shapiro-Wilk test statistics: {sw[0]:.4f}, p-value: {sw[1]:.4f}')
    print(f'Kolmogorov-Smirnov test statistics: {ks.statistic:.4f}, p-value: {ks.pvalue:.4f}')
    print(f'Anderson-Darling test statistics: {ad.statistic:.4f}, 5% critical value: {ad.critical_values[2]:.4f}')
    print('If the returned Anderson Draling statistic is larger than the critical value, then for the 5% significance level, the null hypothesis that the data come from the Normal distribution should be rejected. ')
    
test_normality_of_residuals(lm_sm)
#Function to plot standardized residuals vs Leverage and cook's distance for outlier detection

def influential_outlier_test(model,top_influencing_obs_count):
    
    influence = model.get_influence()

#leverage (hat values)
    leverage = influence.hat_matrix_diag

#When cases are outside of the Cook’s distance (meaning they have high Cook’s distance scores), 
#the cases are influential to the regression results. The regression results will be altered if we exclude those cases.
    cooks_d = influence.cooks_distance

#standardized residuals= (Residual/STD of Residuals)
    standardized_residuals = influence.resid_studentized_internal

#studentized residuals
    studentized_residuals = influence.resid_studentized_external 
    
    
    plot_lm = plt.figure(figsize=(15,6))
    plt.scatter(leverage, standardized_residuals, alpha=0.5)
    sns.regplot(leverage, standardized_residuals,scatter=False,ci=False,lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plot_lm.axes[0].set_xlim(0, max(leverage)+0.01)
    plot_lm.axes[0].set_ylim(-10, 6)
    plot_lm.axes[0].set_title('Standardized Residuals vs Leverage',fontsize=16)
    plot_lm.axes[0].set_xlabel('Leverage',fontsize=13)
    plot_lm.axes[0].set_ylabel('Standardized Residuals',fontsize=13);

    # annotations- #annotating index position of the top n cook's D points 
    
    leverage_top_n_obs = np.flip(np.argsort(cooks_d[0]), 0)[:top_influencing_obs_count]  
    
    for i in leverage_top_n_obs:
        plot_lm.axes[0].annotate(i,xy=(leverage[i],studentized_residuals[i])) 
    
    
# shenanigans for cook's distance contours
    def graph(formula, x_range, label=None):
        x = x_range
        y = formula(x)
        plt.plot(x, y, label=label, lw=1, ls='--', color='red')

    p = len(lm_sm.params) # number of model parameters

    graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), np.linspace(0.001, max(leverage), 50),'Cook\'s distance')#cookd= 0.5 line
    plt.legend(loc='upper right');
    

influential_outlier_test(model=lm_sm,top_influencing_obs_count=10)
#Removing obs 24 & 91 from X_train and y_train

X_train_without_outliers=X_train.drop(index=[24,91])
y_train_without_outliers=y_train.drop(index=[24,91])
from sklearn.feature_selection import RFE
import statsmodels.api as sm

n_features_list = list(range(4, 31)) #checking for optimal number of features between 4 to 30
train_adjusted_r2 = []
train_r2 = []
test_r2 = []
train_RMSE=[]
test_RMSE=[]

for n_features in range(4, 31):

    # RFE with n features
    lm = LinearRegression()

    rfe_n = RFE(estimator=lm, n_features_to_select=n_features)  # specifying number of features

    # fit with n features
    rfe_n.fit(X_train_without_outliers, y_train_without_outliers)

    # selecting the features provided by rfe_n
    col_n = X_train_without_outliers.columns[rfe_n.support_] #rfe_n.support_: returns an array with boolean values to indicate whether 
    #an attribute was selected using RFE

    # subsetting training & test data for n selected columns
    X_train_rfe_n = X_train_without_outliers[col_n]
    X_test_rfe_n = X_test[col_n]


    # add a constant to the model
    X_train_rfe_n = sm.add_constant(X_train_rfe_n,has_constant='add')


    X_test_rfe_n = sm.add_constant(X_test_rfe_n, has_constant='add')

    
    
    # fitting the model with n featues
    lm_n = sm.OLS(y_train_without_outliers, X_train_rfe_n).fit()
    
    
    # # Making predictions
    y_pred_test = lm_n.predict(X_test_rfe_n)
    y_pred_train = lm_n.predict(X_train_rfe_n)
    
    
    #Calculating evaluation metrics
    
    #R-square
    train_adjusted_r2.append(lm_n.rsquared_adj)
    train_r2.append(lm_n.rsquared)
    test_r2.append(r2_score(y_test, y_pred_test))
    
    #RMSE/stan. error
    error_test=y_pred_test-y_test
    error_train=y_pred_train-y_train_without_outliers
    
    test_RMSE.append(((error_test**2).mean())**0.5)
    train_RMSE.append(((error_train**2).mean())**0.5)
# plotting r2 and RMSE against n_features
fig,ax=plt.subplots(2,1,figsize=(13, 9))
plt.subplots_adjust(hspace = 0.3)
ax[0].plot(n_features_list, train_r2,'b', label="r2_train data")
ax[0].plot(n_features_list, test_r2,'g', label="r2_test data")
ax[0].set_xlabel('Features Count',fontsize=13)
ax[0].set_ylabel('R-Squared',fontsize=13)
ax[0].set_title('R-Squared by number of features in the model',fontsize=16)

ax[1].plot(n_features_list, train_RMSE, 'b',label="RMSE_train data")
ax[1].plot(n_features_list, test_RMSE, 'g',label="RMSE_test data")
ax[1].set_xlabel('Features Count',fontsize=13)
ax[1].set_ylabel('RMSE',fontsize=13)
ax[1].set_title('RMSE by number of features in the model',fontsize=16)

ax[0].legend(loc='upper left')
ax[1].legend(loc='upper right')
plt.show()
RMSE_test_dividedby_train = [i / j for i, j in zip(test_RMSE, train_RMSE)]
RMSE_test_dividedby_train
# Remove the outliers from X & Y and create new dfs for Cross Validation

X_new_cross_v = X_scaled.drop(index=[24,91]) # DF for K fold cross validation (cv)
y_new_cross_v = y.drop(index=[24,91]) 
#Resetting index as we need to use K-fold and thus index needs to be in proper order
X_new_cross_v.reset_index(inplace=True, drop=True)
X_new_cross_v.head(20)
y_new_cross_v.reset_index(drop=True,inplace=True)
y_new_cross_v.head(20)
print(X_new_cross_v.shape,y_new_cross_v.shape)
from sklearn.model_selection import KFold

K=5 #using 5 folds
kf = KFold(n_splits=K, shuffle=True, random_state=42)
   
for n_features in range(5,31):
    
    train_RMSE = []
    test_RMSE = []
    train_r2=[]
    test_r2=[]
    
    for train, test in kf.split(X_new_cross_v):
        
        lm = LinearRegression()
        
        rfe_n = RFE(estimator=lm, n_features_to_select=n_features)
        
        rfe_n.fit(X_new_cross_v.loc[train],y_new_cross_v[train])
        
        y_pred_train=rfe_n.predict(X_new_cross_v.loc[train])
        y_pred_test=rfe_n.predict(X_new_cross_v.loc[test])
        
        #R-square
        train_r2.append(r2_score(y_pred_train , y_new_cross_v[train]))
        test_r2.append(r2_score(y_pred_test , y_new_cross_v[test]))
        
        #Error
        error_train = y_pred_train - y_new_cross_v[train]
        error_test = y_pred_test - y_new_cross_v[test]
        rmse_train=((error_train**2).mean())**0.5
        rmse_test=((error_test**2).mean())**0.5
        
        train_RMSE.append(rmse_train)
        test_RMSE.append(rmse_test)
        
    test_times_train=np.mean(test_RMSE)/np.mean(train_RMSE)
         # generate report
    print('n_features:{:1} |train_R2:{:2} |test_R2:{:3} |mean(rmse_train):{:4} |mean(rmse_test):{:5} |RMSE(test/train):{}'.
          format(n_features, round(np.mean(train_r2),4), round(np.mean(test_r2),4),
                 round(np.mean(train_RMSE),0),
                 round(np.mean(test_RMSE),0),round(test_times_train,2)))
import statsmodels.api as sm

lm = LinearRegression()

rfe = RFE(estimator=lm, n_features_to_select=10)

rfe.fit(X_new_cross_v, y_new_cross_v)

col= X_new_cross_v.columns[rfe.support_] 

X_final=X_new_cross_v[col]

X_final= sm.add_constant(X_final,has_constant='add')

lm_sm=sm.OLS(y_new_cross_v,X_final).fit()

print(lm_sm.summary())

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

def get_VIFs(X_df): #X_df = X_train normally, in this model X=X_final
    X_df = add_constant(X_df)
    vifs = pd.Series(
        [1 / (1. - OLS(X_df[col].values, X_df.loc[:, X_df.columns != col].values).fit().rsquared) for col in X_df],
        index=X_df.columns,
        name='VIF'
    )
    return vifs

get_VIFs(X_final)
X_final =X_final.loc[:,X_final.columns !='company_peugeot']
X_final.head()
get_VIFs(X_final)
X_final =X_final.loc[:,X_final.columns !='enginesize']
X_final.head()
get_VIFs(X_final)
lm_sm=sm.OLS(y_new_cross_v,X_final).fit()

y_predictions=lm_sm.predict(X_final)


#Standard error/RMSE
error=y_predictions-y_new_cross_v

print('RMSE is: {}'.format(((error**2).mean())**0.5))
print(lm_sm.summary())
X_final =X_final.loc[:,X_final.columns !='enginetype_ohcv']
X_final.head()
get_VIFs(X_final)
lm_sm=sm.OLS(y_new_cross_v,X_final).fit()

y_predictions=lm_sm.predict(X_final)


#Standard error/RMSE
error=y_predictions-y_new_cross_v

print('RMSE is: {}'.format(((error**2).mean())**0.5))
print(lm_sm.summary())
# lets drop 'enginetype_rotor' as well and see the predictions again.
X_final =X_final.loc[:,X_final.columns !='enginetype_rotor']
X_final.head()
get_VIFs(X_final)
lm_sm=sm.OLS(y_new_cross_v,X_final).fit()

y_predictions=lm_sm.predict(X_final)


#Standard error/RMSE
error=y_predictions-y_new_cross_v

print('RMSE is: {}'.format(((error**2).mean())**0.5))
print(lm_sm.summary())
# lets drop 'boreratio' which has a p-value of 0.068.
X_final =X_final.loc[:,X_final.columns !='boreratio']
X_final.head()

get_VIFs(X_final)
# predict again
lm_sm=sm.OLS(y_new_cross_v,X_final).fit()

y_predictions=lm_sm.predict(X_final)


#Standard error/RMSE
error=y_predictions-y_new_cross_v

print('RMSE is: {}'.format(((error**2).mean())**0.5))
print(lm_sm.summary())
fig, ax=plt.subplots(figsize=(15,6))
sns.lineplot(x=y_new_cross_v.index,y=y_new_cross_v,label='Actuals',color='blue',ax=ax)
sns.lineplot(x=y_new_cross_v.index,y=y_predictions,label='Predictions',color='red',ax=ax)
ax.set_title('Car Price: Predicted Vs Actuals', fontsize=16)
ax.set_ylabel('Car Price',fontsize=13)
test_linearity_of_model(lm_sm,y_new_cross_v)
test_homoscedasticity_of_model(lm_sm)
influential_outlier_test(model=lm_sm,top_influencing_obs_count=10)
test_normality_of_residuals(lm_sm)
print(f'The R2 score for y_test and y_test_pred is: {round(r2_score(y_predictions , y_new_cross_v),2)}')