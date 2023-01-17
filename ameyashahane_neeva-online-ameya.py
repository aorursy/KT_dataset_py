# KAGGLE Version of Neeva submission; please modify paths specified under chunk[4] with variables shared_articles_df and user_interacts_df by absolute path. 
# Thanks!


# basic libraries
from scipy import stats
import math
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
#https://www.kaggle.com/getting-started/25930 import kaggle dataset;
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 
import os
for dirname, _, filenames in os.walk('/kaggle/input'): # optional but good for automation
    for filename in filenames:
        print(os.path.join(dirname, filename))


#path_1 = '../input/articles-sharing-reading-from-cit-deskdrop/shared_articles.csv'
#path_2 = '../input/articles-sharing-reading-from-cit-deskdrop/users_interactions.csv'
shared_articles_df = pd.read_csv('../input/articles-sharing-reading-from-cit-deskdrop/shared_articles.csv') #locally: /home/ameya/Downloads/shared_articles.csv
shared_articles_df=shared_articles_df.fillna(0) # get rid of bad data
user_interacts_df = pd.read_csv('../input/articles-sharing-reading-from-cit-deskdrop/users_interactions.csv') #locally: /home/ameya/Downloads/users_interactions.csv
user_interacts_df=user_interacts_df.fillna(0) # get rid of bad data
combined_df = pd.merge(shared_articles_df, user_interacts_df, on = 'contentId', right_index = True) # pandas sql join on contentId based on documentation
combined_df=combined_df.fillna(0)  # get rid of bad data to handle errors in linear regression modelling with NA values
combined_df


df= user_interacts_df
df['RUNNER'] = 1 #init as 1
my_data = df.groupby(['contentId','eventType'])['RUNNER'].sum().reset_index() #sum function
group_events = my_data.pivot_table('RUNNER', ['contentId'], 'eventType')
group_events = group_events.fillna(0) # avoid linear regression errors later
def label(vect):
    VIRALITY = (1* vect['VIEW']) + (4*vect['LIKE']) + (10 * vect['COMMENT CREATED']) +( 25*vect['FOLLOW'] )+ (100*vect['BOOKMARK'])  # evaluation metric
    return (VIRALITY)

group_events['label'] = group_events.apply (lambda vect: label(vect), axis=1)


from sklearn.model_selection import train_test_split
def label(vect): # data metrics for virality
    VIRALITY = (1* vect['VIEW']) + (4*vect['LIKE']) + (10 * vect['COMMENT CREATED']) +( 25*vect['FOLLOW'] )+ (100*vect['BOOKMARK'])  # evaluation metric
    return (VIRALITY)
df_events = combined_df.groupby(['contentId','eventType_x']).sum() # grouped evaluation for combined_df
df_events['label'] = group_events.apply (lambda vect: label(vect), axis=1) # lambda type function; 
df_events=df_events.fillna(0)


from sklearn.model_selection import train_test_split

train, test = train_test_split(group_events, test_size=0.2)

from sklearn.linear_model import LinearRegression  # MUST include imports here to prevent non callable error.
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

train_X, train_Y = train.drop('label',axis = 1), train['label']
test_X, test_Y = test.drop('label',axis = 1), test['label']
#X_train, X_test = train_test_split(df_events, test_size=0.2)
LinearRegression = LinearRegression() 
LinearRegression.fit(train_X,train_Y)
score= LinearRegression.score(train_X,train_Y) # R^2 
pred_test = LinearRegression.predict(test_X) # predictive functionality
intercept = LinearRegression.intercept_ # y=mx+b simple linear regression
coeff = LinearRegression.coef_
print(intercept) # this is intercept
print(coeff) # this is coefficient
print(score) # 100% # R^2 relation of model highly suggesting overfit of model !!
print("Linear Regression Overfitted Model Done")
#print(pred_test)