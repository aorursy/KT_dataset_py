# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#reading the data
df= pd.read_csv('/kaggle/input/imdb-5000-movie-dataset/movie_metadata.csv')
#importing all the necessary libraries
import matplotlib.pyplot as plt
import seaborn           as sns
import statsmodels.api   as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import scipy.stats as stats

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.formula.api import ols


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

pd.options.display.max_columns=None

pd.options.display.max_rows=None
import pandas_profiling

df.profile_report()
df.info()
df.shape
df.corr()
# dropping all the un-necessary columns
df = df.drop(['color', 'movie_imdb_link', 'movie_title', 'plot_keywords', 'director_name','actor_1_name', 'actor_2_name', 'actor_3_name', 'country','language', 'genres', 'title_year'], axis=1)
df.shape
#content_rating is a categorical column so let's replace missing value for this column with its' mode.

content_rating_mode= df['content_rating'].mode()
df.content_rating.fillna(content_rating_mode[0],inplace=True)



# For All numerical columns, we will replace the missing values with their medians.

medianlist=['actor_2_facebook_likes','actor_1_facebook_likes','num_user_for_reviews','director_facebook_likes',
            'gross','duration', 'num_critic_for_reviews','budget','actor_3_facebook_likes',
            'num_critic_for_reviews','aspect_ratio','facenumber_in_poster']


def median(i):
    median= df[i].median()
    i = df[i].fillna(median)
    return i



for i in medianlist:
    df[i]= median(i)
df.skew()
# Creating dummies for content_rating.
top_10_CR=[x for x in df.content_rating.value_counts().sort_values(ascending=False).head(10).index]

for label in top_10_CR:
    df["content_rating"+"_"+label]=np.where(label==df["content_rating"],1,0)

#Dropping the original column after 
df= df.drop('content_rating',axis=1)
df.head(1)
df1= df.copy()
for i in df:
    df[i] = df[i].map(lambda i: np.log1p(i) ) 
df.head()
x=df.drop('imdb_score', axis=1)
Y=df['imdb_score']
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
pd.DataFrame()
X= pd.DataFrame(sc.fit_transform(x), columns=x.columns)
X.head(1)
df.head(1)

# train test split
x_train,x_test,y_train,y_test = train_test_split(X ,Y,test_size = 0.3,random_state = 1)
rfc= RandomForestRegressor(random_state=1)
hyper={'n_estimators': np.arange(1,50)}

rfc_grid=GridSearchCV(estimator= rfc, param_grid=hyper, verbose=True)

rfc_grid.fit(x_train,y_train)

rfc_grid.best_params_
knn_param= {'n_neighbors': np.arange(3,30), 'weights':['uniform', 'distance']}
knn = KNeighborsRegressor()
knn_grid= GridSearchCV(knn, knn_param, cv=5, scoring='neg_mean_squared_error')
knn_grid.fit(x_train, y_train)

knn_grid.best_params_
dt= DecisionTreeRegressor(random_state=1)
dt_params= {'max_depth': np.arange(1,50), 'min_samples_leaf': np.arange(2,15)} #2,15 not too high, not too low

GS_dt= GridSearchCV(dt, dt_params, cv=5, scoring='neg_mean_squared_error')

GS_dt.fit(x_train, y_train)

GS_dt.best_params_
from sklearn import ensemble

RF= RandomForestRegressor(**rfc_grid.best_params_, random_state=1)
ensemble_params= {'n_estimators': np.arange(1,20)} 
AB_RF= ensemble.AdaBoostRegressor(base_estimator=RF ,random_state=1)

GS_AB_RF = GridSearchCV(AB_RF, ensemble_params, cv=5, scoring='neg_mean_squared_error')

GS_AB_RF.fit(x_train, y_train)
GS_AB_RF.best_params_

#declare the models
lr  = LinearRegression()
RF  = RandomForestRegressor(n_estimators= 48, random_state=1)
knn = KNeighborsRegressor(n_neighbors= 10, weights= 'distance')
dt  = DecisionTreeRegressor(max_depth= 6, min_samples_leaf= 13)
bgc =ensemble.BaggingRegressor(base_estimator=lr)
AB_RF= ensemble.AdaBoostRegressor(**GS_AB_RF.best_params_, base_estimator=RF, random_state=1)
gb  =ensemble.GradientBoostingRegressor()

#create a list of models
models=[lr,RF ,knn, dt, bgc,AB_RF, gb]

def score_model(xtrain,ytrain,xtest,ytest):
    mod_columns=[]
    mod=pd.DataFrame(columns=mod_columns)
    i=0
    #read model one by one
    for model in models:
        model.fit(xtrain,ytrain)
        y_pred=model.predict(xtest)
        
        
        
        
        #compute metrics
        train_accuracy=model.score(xtrain,ytrain)
        test_accuracy=model.score(xtest,ytest)
        
        #insert in dataframe
        mod.loc[i,"Model_Name"]=model.__class__.__name__
        mod.loc[i,"Train_Accuracy"]=round(train_accuracy,2)
        mod.loc[i,"Test_Accuracy"]=round(test_accuracy,2)
        
        i+=1

    
    return(mod)
report=score_model(x_train,y_train,x_test,y_test)
report
# As we had inferred above, we can see that linear regression is not giving good results.
# We can conclude that gradient boosting Regressor is the best model as it has good train and test accuracies as compared to other models.