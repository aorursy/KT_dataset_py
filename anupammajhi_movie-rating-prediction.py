import pandas as pd

import numpy as np



# Increase visible rows and columns

pd.options.display.max_columns = 100

pd.options.display.max_rows = 100
# Importing movie_metadata.csv

moviedata = pd.read_csv('../input/movie_metadata.csv',encoding = "ISO-8859-1")
moviedata.head()
# information about dataset

moviedata.info(max_cols=300)
moviedata.describe(include='all')
# First lets drop the columns which is not going to be useful for our analysis



# the movie link and name are of no use for analysis



columns_nouse = ['movie_title','movie_imdb_link']



moviedata_1 = moviedata.drop(columns_nouse,axis=1)



moviedata_1.columns
# Creating dummy variables for genres



genres_dummy = moviedata_1['genres'].str.get_dummies('|')

genres_dummy.columns
# Removing columns that make no sense



genres_dummy = genres_dummy.drop(['-', 'A', 'B', 'C',

       'D', 'F', 'H', 'M', 'S', 'T',  'W', 'a', 'c', 'd', 'e',

       'g', 'h', 'i', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'y'],axis=1)



genres_dummy.columns
for colname in genres_dummy.columns:

    genres_dummy = genres_dummy.rename(columns={colname:("genre_" + colname)})
moviedata_1 = pd.concat([moviedata_1,genres_dummy],axis=1)



moviedata_1.describe(include='all')
# We can see several columns with Null/NA Values.



# Get columns with NA by finding percentage

def get_na_percentage(df):

    NA_perc = pd.DataFrame(df.isna().sum() / len(df) * 100)

    return NA_perc.sort_values(by=[0], ascending=False)



get_na_percentage(moviedata_1)
# For some numerical features we can impute the NA to 0



columns_impute_zero = ['facenumber_in_poster','actor_1_facebook_likes',

                       'actor_2_facebook_likes','num_critic_for_reviews',

                       'director_facebook_likes','num_user_for_reviews',

                       'actor_3_facebook_likes']



for colname in columns_impute_zero:

    moviedata_1[colname].fillna(0,inplace=True)

    

get_na_percentage(moviedata_1)
# We will replace the NA with a vlaue "Unknown" for categorical fields like actor names and director names



columns_impute_unknown = ['actor_1_name','actor_2_name','actor_3_name','country','language','color','content_rating',

                         'plot_keywords','director_name']



for colname in columns_impute_unknown:

    moviedata_1[colname].fillna('Unknown',inplace=True)

    

get_na_percentage(moviedata_1)
# We will drop the rows where NA values are present for few columns

columns_remove_na_rows = ['aspect_ratio','duration','title_year']





moviedata_1.dropna(subset = columns_remove_na_rows , inplace=True)

    

get_na_percentage(moviedata_1)
median_budget = np.nanmedian(moviedata_1['budget'])

moviedata_1['budget'].fillna(median_budget,inplace=True)



get_na_percentage(moviedata_1)
import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



# To ignore warnings

import warnings

warnings.filterwarnings("ignore")



def plot_univariate_num(dfname,colname):

    plt.figure(figsize=(18,10))

    plt.xlabel('x',fontsize=15)

    plt.ylabel('Y ',fontsize=15)

    plt.xticks(rotation=50,fontsize=12,ha="right")

    plt.yticks(fontsize=12,ha="right")

    ax = sns.distplot(dfname[colname].dropna())

    plt.show()
def plot_bivariate_num(colname,colname2):

    

    def get_df(dfname,bins=10):

        df1 = pd.DataFrame(dfname[colname2]).reset_index(drop=True)

        df2 = pd.DataFrame(pd.cut(dfname[colname],bins=bins)).reset_index(drop=True)

        df = pd.concat([df2,df1],axis=1)

        return df

    

    #return get_df(df_top_3_purpose_debt_consolidation)

    plt.figure(figsize = [18,8])

    

    plot_1_data = get_df(moviedata_1)

    plot_1 = sns.barplot(x=colname,y=colname2,data=plot_1_data)

    plt.xlabel(colname,fontsize=15)

    plt.title((colname.upper() + " vs. " + colname2.upper()),fontsize=15)

    plt.ylabel(colname2,fontsize=15)

    plt.xticks(rotation=50,fontsize=12,ha="right")

    plt.yticks(fontsize=12,ha="right")



    plt.show()

    
# Top Actors



moviedata_1[['actor_1_name','actor_1_facebook_likes']].groupby(['actor_1_name']).mean().sort_values(by=['actor_1_facebook_likes'],ascending = False)
moviedata_1.describe()
# Detecting Outliers



def get_outlier_percentage(x):

    Q1 = x.quantile(0.25)

    Q3 = x.quantile(0.75)

    

    IQR = Q3 - Q1

    

    outlier_min = Q1 - 1.5 * IQR

    outlier_max = Q3 + 1.5 * IQR

    

    return {'%' : round(((x <= outlier_min) | (x >= outlier_max)).sum() / len(x) * 100,2), 'min' : round(outlier_min,2) ,'max': round(outlier_max,2)}





def plot_boxplot(x):

    fig = plt.figure(figsize = [18,2])

    ax= sns.boxplot(x)

    plt.show()
# director facebook likes

print(pd.DataFrame(moviedata_1['director_facebook_likes']).apply(get_outlier_percentage,axis=0))

plot_boxplot(moviedata_1['director_facebook_likes'])
# Capping director facebook likes

moviedata_1['director_facebook_likes'] = moviedata_1['director_facebook_likes'].map(lambda x: 178 if x > 178 else x)
# actor1 facebook likes

print(pd.DataFrame(moviedata_1['actor_1_facebook_likes']).apply(get_outlier_percentage,axis=0))

plot_boxplot(moviedata_1['actor_1_facebook_likes'])



# actor2 facebook likes

print(pd.DataFrame(moviedata_1['actor_2_facebook_likes']).apply(get_outlier_percentage,axis=0))

plot_boxplot(moviedata_1['actor_2_facebook_likes'])



# actor3 facebook likes

print(pd.DataFrame(moviedata_1['actor_3_facebook_likes']).apply(get_outlier_percentage,axis=0))

plot_boxplot(moviedata_1['actor_3_facebook_likes'])



# cast facebook likes

print(pd.DataFrame(moviedata_1['cast_total_facebook_likes']).apply(get_outlier_percentage,axis=0))

plot_boxplot(moviedata_1['cast_total_facebook_likes'])

# Capping actor facebook likes

moviedata_1['actor_1_facebook_likes'] = moviedata_1['actor_1_facebook_likes'].map(lambda x: 1248 if x > 1248 else x)

moviedata_1['actor_2_facebook_likes'] = moviedata_1['actor_2_facebook_likes'].map(lambda x: 575 if x > 575 else x)

moviedata_1['actor_3_facebook_likes'] = moviedata_1['actor_3_facebook_likes'].map(lambda x: 326 if x > 326 else x)

moviedata_1['cast_total_facebook_likes'] = moviedata_1['cast_total_facebook_likes'].map(lambda x: 2686 if x > 2686 else x)
# movie duration

print(pd.DataFrame(moviedata_1['duration']).apply(get_outlier_percentage,axis=0))

plot_boxplot(moviedata_1['duration'])



# capping duration

moviedata_1['duration'] = moviedata_1['duration'].map(lambda x: 1.95 if x > 1.95 else x)
# movie facebook like 

print(pd.DataFrame(moviedata_1['movie_facebook_likes']).apply(get_outlier_percentage,axis=0))

plot_boxplot(moviedata_1['movie_facebook_likes'])



# movie facebook like  capping

moviedata_1['movie_facebook_likes'] = moviedata_1['movie_facebook_likes'].map(lambda x: 687 if x > 687 else x)
# reviews and votes



print(pd.DataFrame(moviedata_1['num_voted_users']).apply(get_outlier_percentage,axis=0))

plot_boxplot(moviedata_1['num_voted_users'])

print(pd.DataFrame(moviedata_1['num_user_for_reviews']).apply(get_outlier_percentage,axis=0))

plot_boxplot(moviedata_1['num_user_for_reviews'])

print(pd.DataFrame(moviedata_1['num_critic_for_reviews']).apply(get_outlier_percentage,axis=0))

plot_boxplot(moviedata_1['num_critic_for_reviews'])



# reviews and votes capping

moviedata_1['num_voted_users'] = moviedata_1['num_voted_users'].map(lambda x: 283674.0 if x > 283674.0 else x)

moviedata_1['num_user_for_reviews'] = moviedata_1['num_user_for_reviews'].map(lambda x: 820.88 if x > 820.88 else x)

moviedata_1['num_critic_for_reviews'] = moviedata_1['num_critic_for_reviews'].map(lambda x: 449.0 if x > 449.0 else x)
# facenumber_in_poster



print(pd.DataFrame(moviedata_1['facenumber_in_poster']).apply(get_outlier_percentage,axis=0))

plot_boxplot(moviedata_1['facenumber_in_poster'])



# capping

moviedata_1['facenumber_in_poster'] = moviedata_1['facenumber_in_poster'].map(lambda x: 5 if x > 5 else x)
# gross and budget



# facenumber_in_poster



print("Gross",pd.DataFrame(moviedata_1['gross']).apply(get_outlier_percentage,axis=0)['gross'])

plot_boxplot(moviedata_1['gross'])



print("Budget",pd.DataFrame(moviedata_1['budget']).apply(get_outlier_percentage,axis=0)['budget'])

plot_boxplot(moviedata_1['budget'])



# capping

moviedata_1['gross'] = moviedata_1['gross'].map(lambda x: 449955182.5 if x > 449955182.5 else x)

moviedata_1['budget'] = moviedata_1['budget'].map(lambda x: 113000000.0 if x > 113000000.0 else x)
cor_1 = moviedata_1.corr()

plt.figure(figsize=(20,20))

sns.heatmap(cor_1,annot = True)
# To have a simpler view of high correlation, we will unstack the pairs



# To check correlation either positive or negative, we will use absolute values



cor_unstacked = cor_1.unstack()



cor_unstacked[cor_unstacked < 1].sort_values(ascending = False)

# Number of User Reviews on Movies



# We had seen above that:

# mean vote is about 305

# median vote is about 175 

# whereas the max goes upto 6000

# there is a standard deviation of about 425



plot_univariate_num(moviedata_1,'num_user_for_reviews')



# We can observe here that most movies have very less number of reviews compared to some popular movies which have revies close to 1000s



# there is surge at end due to capped outliers
# Number of User Votes on Movies



# We had seen above that:

# mean vote is about 98000

# median vote is about 41000 

# whereas the max goes upto 2 million

# there is a standard deviation of about 160000



plot_univariate_num(moviedata_1,'num_voted_users')



# We can observe here that most movies have very less number of votes compared to some popular movies which have votes in huge numbers



# there is surge at end due to capped outliers
# imdb score



plot_univariate_num(moviedata_1,'imdb_score')



# Looking at the dependent variable here

# the IMBD ratings seem to be quite normally distributed with spikes seen at even number probably because people rate in absolute numbers
# Gross box office collection



plot_univariate_num(moviedata_1,'gross')



# the gross collection is towards lower side for most movies, also displaying few movies have got exceptionally high gross collection



# there is surge at end due to capped outliers
# num_critic_for_reviews



plot_univariate_num(moviedata_1,'num_critic_for_reviews')



# unlike user reviews, critic reviews are quite wide spread towards all films, this could be a good measure to use



# there is surge at end due to capped outliers
y = moviedata_1['imdb_score']



g1 = moviedata_1['actor_1_facebook_likes']

g2 = moviedata_1['actor_2_facebook_likes']

g3 = moviedata_1['actor_3_facebook_likes']

 

data = (g1, g2, g3)

colors = ("red", "green", "blue")

groups = ("Actor1", "Actor2", "Actor3") 

 

# Create plot



fig = plt.figure(figsize = [18,8])

ax = fig.add_subplot(1, 1, 1, facecolor="1.0")

 

for data, color, group in zip(data, colors, groups):

    x = data

    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)





plt.title('Actors and IMDB Score')

plt.legend(loc=2)

plt.show()
# The above plot shows that the actor likes is pretty much concentrated and shows high IMDB scores
y = moviedata_1['imdb_score']



g2 = moviedata_1['num_user_for_reviews']

g3 = moviedata_1['num_critic_for_reviews']

 

data = (g2, g3)

colors = ("red", "green")

groups = ("User Review", "Critic Review") 

 

# Create plot



fig = plt.figure(figsize = [18,8])

ax = fig.add_subplot(1, 1, 1, facecolor="1.0")

 

for data, color, group in zip(data, colors, groups):

    x = data

    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)





plt.title('Review Count and IMDB Score')

plt.legend(loc=2)

plt.show()
plot_bivariate_num('actor_1_facebook_likes','imdb_score')



#There doesn't seem to be a very high relationship between imdb score and actor facebook likes
    

plot_bivariate_num('budget','gross')



# Higher budget films tend to make higher box office gross income
plot_bivariate_num('actor_1_facebook_likes','cast_total_facebook_likes')  



# As expected from the high correlation, the plot looks uniform
plot_bivariate_num('num_critic_for_reviews','imdb_score')



# Higher rated movies tend to get more critic reviews
plot_bivariate_num('budget','imdb_score')



# Budget doesn't seem to have a very high effect on imdb score
# we will remove few columns which are not useful for linear regression modelling



# plot keyword has too many unique entries, hence ignoring



# Since we have seen gross has a very high correlation with budget for example, hence we will drop the gross



columns_nouse = ['actor_1_name','actor_2_name','actor_3_name','director_name','plot_keywords','gross']





moviedata_1 = moviedata_1.drop(columns_nouse,axis=1)



moviedata_1.columns
# We will create the categorical features as dummy variable



moviedata_1['language'].value_counts()
moviedata_1['language'] = moviedata_1['language'].apply(lambda x : 'Other' if x not in ['English','French','Spanish','Mandarin','Hindi','Japanese','German','Cantonese'] else x)

moviedata_1['language'].value_counts()
language_dummy = moviedata_1['language'].str.get_dummies()

language_dummy = language_dummy.loc[:, language_dummy.columns != 'Other']
moviedata_1['country'].value_counts()
moviedata_1['country'] = moviedata_1['country'].apply(lambda x : 'Other' if x not in ['USA','UK','France','Canada','Germany','Australia',

'China','Spain','Japan','India','Hong Kong','Italy'] else x)

moviedata_1['country'].value_counts()
country_dummy = moviedata_1['country'].str.get_dummies()

country_dummy = country_dummy.loc[:, country_dummy.columns != 'Other']
moviedata_1['color'].value_counts()
color_dummy = moviedata_1['color'].str.get_dummies()

color_dummy = color_dummy.loc[:, color_dummy.columns != 'Unknown']
moviedata_1['content_rating'].value_counts()
content_rating_dummy = moviedata_1['content_rating'].str.get_dummies()

content_rating_dummy = content_rating_dummy.loc[:, content_rating_dummy.columns != 'Unknown']
for colname in language_dummy.columns:

    language_dummy = language_dummy.rename(columns={colname:("langauge_" + colname)})

    

for colname in country_dummy.columns:

    country_dummy = country_dummy.rename(columns={colname:("country_" + colname)})

    

for colname in content_rating_dummy.columns:

    content_rating_dummy = content_rating_dummy.rename(columns={colname:("Rating_" + colname)})



for colname in color_dummy.columns:

    color_dummy = color_dummy.rename(columns={colname:("color_" + colname)})
moviedata_1.columns
# Adding the dummy variables and removing the categorical variables for which dummy created



moviedata_1 = pd.concat([moviedata_1,language_dummy,country_dummy,content_rating_dummy,color_dummy],axis=1)



columns_nouse = ['language', 'country', 'genres', 'color', 'content_rating']



moviedata_1 = moviedata_1.drop(columns_nouse,axis=1)



moviedata_1.describe(include='all')
# Putting feature variable to X

X = moviedata_1.drop(['imdb_score'],axis=1)



# Putting response variable to y

y = moviedata_1['imdb_score']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 , random_state=100)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()



# fit the model to the training data

lm.fit(X_train,y_train)
# print the intercept

print(lm.intercept_)
# Let's see the coefficient

coeff_df = pd.DataFrame(lm.coef_,X_test.columns,columns=['Coefficient'])

coeff_df
# Making predictions using the model



y_pred = lm.predict(X_test)



y_pred_lm_default = y_pred
#Error Terms

from sklearn.metrics import mean_squared_error, r2_score



mse = mean_squared_error(y_test, y_pred)

r_squared = r2_score(y_test, y_pred)



print('Mean_Squared_Error :' ,mse)

print('r_square_value :',r_squared)
import statsmodels.api as sm



X_train_sm = X_train

X_train_sm = sm.add_constant(X_train_sm)



lm_1 = sm.OLS(y_train,X_train_sm).fit()



#Let's see the summary of our linear model

print(lm_1.summary())
# UDF for calculating vif value

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
vif_cal(input_data=moviedata_1, dependent_col="imdb_score")



# we can see very high VIFs which denote that there is very high multicolliniarity in existance
# Using RFE for feature selection

from sklearn.feature_selection import RFE



# Running RFE with number of the variable = 20

lm_2 = LinearRegression()

rfe = RFE(lm_2, 20)             

rfe = rfe.fit(X_train, y_train)
# Creating X_test dataframe with RFE selected variables

col_2 = X_train.columns[rfe.support_]

print(col_2)

X_train_rfe = X_train[col_2]
lm_rfe = sm.OLS(y_train,X_train_rfe).fit()

print(lm_rfe.summary())



# We can see the Adjusted R-squared has now improved a lot
col_rfe = list(col_2.values)

col_rfe.append('imdb_score')



vif_cal(input_data=moviedata_1[col_rfe], dependent_col="imdb_score")



# we still have some high VIFs but it is much better than before
# Making Prediction



# Now let's use our model to make predictions.



# Creating X_test_6 dataframe by dropping variables from X_test

X_test_rfe = X_test[col_2]



# Making predictions

y_pred = lm_rfe.predict(X_test_rfe)

y_pred_lm_rfe = y_pred

# a visual inspection of IMDB ratings



pd.concat([y_pred,y_test],axis=1)
# Actual and Predicted



c = [i for i in range(1,301,1)] # generating index 

fig = plt.figure(figsize=(18,10)) 

plt.plot(c,y_test.head(300), color="blue", linewidth=2.5, linestyle="-") #Plotting Actual

plt.plot(c,y_pred.head(300), color="red",  linewidth=2.5, linestyle="-") #Plotting predicted

fig.suptitle('Actual vs Predicted', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                               # X-label

plt.ylabel('IMDB score', fontsize=16)                       # Y-label
# Plotting the error terms to understand the distribution.



fig = plt.figure(figsize=(18,10))

sns.distplot((y_test-y_pred),bins=50)

fig.suptitle('Error Terms', fontsize=20)                  # Plot heading 

plt.xlabel('y_test - y_pred', fontsize=18)                  # X-label

plt.ylabel('Index', fontsize=16)                          # Y-label



from sklearn.linear_model import Lasso



lr = Lasso(alpha=1e-10,normalize=True, max_iter=1e5)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

y_pred = pd.Series(y_pred)



y_test_0 = y_test.reset_index()

y_test_0 = y_test_0['imdb_score']



y_pred_lasso = y_pred



pd.concat([y_pred,y_test_0],axis=1)  
mse = mean_squared_error(y_test_0, y_pred)

r_squared = r2_score(y_test_0, y_pred)



print('Mean_Squared_Error :' ,mse)

print('r_square_value :',r_squared)
# Actual and Predicted



c = [i for i in range(1,301,1)] # generating index 

fig = plt.figure(figsize=(18,10)) 

plt.plot(c,y_test_0.head(300), color="blue", linewidth=2.5, linestyle="-") #Plotting Actual

plt.plot(c,y_pred.head(300), color="red",  linewidth=2.5, linestyle="-") #Plotting predicted

fig.suptitle('Actual vs Predicted', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                               # X-label

plt.ylabel('IMDB score', fontsize=16)                       # Y-label
# Plotting the error terms to understand the distribution.



fig = plt.figure(figsize=(18,10))

sns.distplot((y_test_0 - y_pred),bins=50)

fig.suptitle('Error Terms', fontsize=20)                  # Plot heading 

plt.xlabel('y_test - y_pred', fontsize=18)                  # X-label

plt.ylabel('Index', fontsize=16)                          # Y-label

import xgboost as xgb

from xgboost import XGBRegressor



xgr = XGBRegressor()

xgr.fit(X_train, y_train)
y_pred = xgr.predict(X_test)

y_pred = pd.Series(y_pred)



y_pred_XGB_default = y_pred



y_test_0 = y_test.reset_index()

y_test_0 = y_test_0['imdb_score']



pd.concat([y_pred,y_test_0],axis=1)
mse = mean_squared_error(y_test_0, y_pred)

r_squared = r2_score(y_test_0, y_pred)



print('Mean_Squared_Error :' ,mse)

print('r_square_value :',r_squared)
# Actual and Predicted



c = [i for i in range(1,301,1)] # generating index 

fig = plt.figure(figsize=(18,10)) 

plt.plot(c,y_test_0.head(300), color="blue", linewidth=2.5, linestyle="-") #Plotting Actual

plt.plot(c,y_pred.head(300), color="red",  linewidth=2.5, linestyle="-") #Plotting predicted

fig.suptitle('Actual vs Predicted', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                               # X-label

plt.ylabel('IMDB score', fontsize=16)                       # Y-label
# Plotting the error terms to understand the distribution.



fig = plt.figure(figsize=(18,10))

sns.distplot((y_test_0 - y_pred),bins=50)

fig.suptitle('Error Terms', fontsize=20)                  # Plot heading 

plt.xlabel('y_test - y_pred', fontsize=18)                  # X-label

plt.ylabel('Index', fontsize=16)                          # Y-label

from sklearn.model_selection import GridSearchCV



parameters = {'objective':['reg:linear'],

              'learning_rate': [0.03, 0.05, 0.1, 0.5, 0.8],

              'max_depth': [5, 6, 8],

              'min_child_weight': [2,4],

              'silent': [1],

              'subsample': [0.5,0.7,1],

              'colsample_bytree': [0.7,1],

              'n_estimators': [200]}



xgb_grid = GridSearchCV(xgr,

                        parameters,

                        cv = 7,

                        n_jobs = 10,

                        verbose=True)



xgb_grid.fit(X_train, y_train)



print(xgb_grid.best_score_)

print(xgb_grid.best_params_)
y_pred = xgb_grid.predict(X_test)

y_pred = pd.Series(y_pred)



y_pred_XGB_grid = y_pred



y_test_0 = y_test.reset_index()

y_test_0 = y_test_0['imdb_score']



pd.concat([y_pred,y_test_0],axis=1)
mse = mean_squared_error(y_test_0, y_pred)

r_squared = r2_score(y_test_0, y_pred)



print('Mean_Squared_Error :' ,mse)

print('r_square_value :',r_squared)
# Actual and Predicted



c = [i for i in range(1,301,1)] # generating index 

fig = plt.figure(figsize=(18,10)) 

plt.plot(c,y_test_0.head(300), color="blue", linewidth=2.5, linestyle="-") #Plotting Actual

plt.plot(c,y_pred.head(300), color="red",  linewidth=2.5, linestyle="-") #Plotting predicted

fig.suptitle('Actual vs Predicted', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                               # X-label

plt.ylabel('IMDB score', fontsize=16)                       # Y-label
# Plotting the error terms to understand the distribution.



fig = plt.figure(figsize=(18,10))

sns.distplot((y_test_0 - y_pred),bins=50)

fig.suptitle('Error Terms', fontsize=20)                  # Plot heading 

plt.xlabel('y_test - y_pred', fontsize=18)                  # X-label

plt.ylabel('Index', fontsize=16)                          # Y-label

predictions = pd.concat([pd.Series(y_pred_lm_default),y_pred_lm_rfe.reset_index()[0],y_pred_lasso,y_pred_XGB_default,y_pred_XGB_grid,y_test_0],axis=1)

predictions.rename(columns={0:"LinearReg",1:"LinearReg_RFE",2:"LassoReg",3:"XGBoost",4:"XGBoost_grid","imdb_score":"Actual IMDB Score"})
y_pred_lm_rfe.reset_index()[0]