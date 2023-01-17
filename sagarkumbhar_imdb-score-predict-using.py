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
df=pd.read_csv('/kaggle/input/imdb-5000-movie-dataset/movie_metadata.csv')

pd.options.display.max_columns=None

df.head()
df.describe()
import numpy             as np

import matplotlib.pyplot as plt

import seaborn           as sns

import statsmodels.api   as sm

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import scale

import scipy.stats as stats

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.linear_model import Ridge, Lasso

from sklearn.model_selection import GridSearchCV

from scipy.spatial import distance

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

import io

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
df.shape
df.drop(['color','director_name','actor_2_name','genres','actor_1_name','movie_title','actor_3_name',

         'plot_keywords','movie_imdb_link','language','country'],axis=1,inplace=True)
df.head()
df.drop('title_year',axis=1,inplace=True)
df.info()
df.content_rating.value_counts()
df.content_rating.isnull().sum()
df.content_rating=df['content_rating'].fillna('R')
df.content_rating.isnull().sum()
from scipy.stats             import ttest_1samp,ttest_ind,chi2_contingency,chisquare, f_oneway, levene, bartlett, mannwhitneyu, normaltest,shapiro
shap_stat, p_val = stats.shapiro(df['imdb_score'])

shap_stat, p_val
# mannwhitneyu for these two:

from statsmodels.formula.api import ols

mod = ols('imdb_score ~ content_rating', data = df).fit()

aov_table = sm.stats.anova_lm(mod, typ=2)

print(aov_table)
dfdummy= pd.get_dummies(df['content_rating'], prefix='content_rating', drop_first=True).reset_index(drop=True)

dfdummy.head()
df= pd.concat([df, dfdummy], axis=1)

df.drop('content_rating',axis=1,inplace=True)
df.head()
df.shape
df.isnull().sum()
for i in df:

    df[i]=df[i].fillna(df[i].median())
df.isnull().sum()
df.head()
df1=df[['num_critic_for_reviews','duration','director_facebook_likes','actor_3_facebook_likes','actor_1_facebook_likes','gross'

       ,'num_voted_users','facenumber_in_poster','cast_total_facebook_likes','num_user_for_reviews','budget','actor_2_facebook_likes'

       ,'aspect_ratio','movie_facebook_likes']]

for i in df1:

    sns.distplot(df[i])

    plt.show()

    print('KDE plot for ',i)
for i in df1:

    sns.boxplot(df[i])

    plt.show()

    print('KDE plot for ',i)
for i in df1:

    df1[i] = df1[i].map(lambda i: np.log1p(i)) 
df1.head()
df['facenumber_in_poster'].value_counts()
plt.figure(figsize=(30,10))

sns.boxplot(x=df['facenumber_in_poster'],y=df['imdb_score'])

plt.plot()
df.drop(['num_critic_for_reviews','duration','director_facebook_likes','actor_3_facebook_likes','actor_1_facebook_likes','gross'

       ,'num_voted_users','cast_total_facebook_likes','num_user_for_reviews','budget','actor_2_facebook_likes'

       ,'aspect_ratio','facenumber_in_poster','movie_facebook_likes'],axis=1,inplace=True)
df_new=pd.concat([df,df1],axis=1)
df_new.head()
df_new.shape
X=df_new.drop('imdb_score',axis=1)

y=df_new['imdb_score']
sc = StandardScaler()

X_scaled = sc.fit_transform(X)



X = pd.DataFrame(X_scaled)

X.head()
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor



from sklearn.ensemble import BaggingRegressor



#declare the models

lr  = LinearRegression()

RF  = RandomForestRegressor(n_estimators= 49, random_state=1)

knn = KNeighborsRegressor(n_neighbors= 10, weights= 'distance')

dt  = DecisionTreeRegressor(max_depth= 6, min_samples_leaf= 13)





#create a list of models

models=[lr,RF ,knn, dt]



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
from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor, VotingRegressor

from sklearn.model_selection import GridSearchCV
LR=LinearRegression()

kNN = KNeighborsRegressor()



DT = DecisionTreeRegressor( random_state=0)

RF = RandomForestRegressor( random_state=0)

GBoost = GradientBoostingRegressor()
models=[]

models.append(('LR',LR))



models.append(('DT',DT))



models.append(('RF',RF))



models.append(('KNN', kNN))





import sklearn.model_selection as model_selection

# evaluate each model in turn

results = []

names = []

for name, model in models:

    kfold = model_selection.KFold(shuffle=True, n_splits=7, random_state=0)

    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error' ) # fit, train, predict

    results.append(np.sqrt(np.abs(cv_results)))    # negative mean squared error 

    names.append(name)

    print('%s: %f (%f)'% (name, np.mean(np.sqrt(np.abs(cv_results))), np.var(np.sqrt(np.abs(cv_results)),ddof=1)))

    

# boxplot algorithm comparision

fig = plt.figure()

fig.suptitle('Algorithm Comparision')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
import statsmodels.formula.api as smf

import sklearn.model_selection as model_selection
#RF

rmse_rf= []

for n_e in np.arange(1,50):

    RF=RandomForestRegressor(n_estimators=n_e,random_state=0,criterion='mae')

    kfold = model_selection.KFold(shuffle=True, n_splits=7, random_state=0)

    mse = model_selection.cross_val_score(RF, X, y, cv=kfold, scoring='neg_mean_squared_error' )

    rmse_rf.append(np.var(np.sqrt(np.abs(mse)), ddof=1))

print(np.argmin(rmse_rf))

#bias error



rmse_be= []

for n_e in np.arange(1,50):

    RF=RandomForestRegressor(n_estimators=n_e,random_state=0,criterion='mae')

    kfold = model_selection.KFold(shuffle=True, n_splits=7, random_state=0)

    mse = model_selection.cross_val_score(RF, X, y, cv=kfold, scoring='neg_mean_squared_error' )

    rmse_be.append(np.mean(np.sqrt(np.abs(mse))))

print(rmse_be)

print("n_estimatr", np.argmin(rmse_be))

print('lowest be', np.min(rmse_be))
RF=RandomForestRegressor(n_estimators=49,random_state=0,criterion='mae')
rmse_GB= []



for n_e in np.arange(1,30):

    GBoost=GradientBoostingRegressor(n_estimators =n_e, random_state=0)

    kfold = model_selection.KFold(shuffle=True, n_splits=7, random_state=0)

    mse = model_selection.cross_val_score(GBoost, X, y, cv=kfold, scoring='neg_mean_squared_error' )

    rmse_GB.append(np.mean(np.sqrt(np.abs(mse))))

print(rmse_GB)

print(np.argmin(rmse_GB))
KNN=KNeighborsRegressor(n_neighbors=3,weights='distance')

DT=DecisionTreeRegressor(max_depth=5,min_samples_leaf=7,criterion='mae', random_state=0)

RF=RandomForestRegressor(n_estimators=41,random_state=0,criterion='mae')





GBoost=GradientBoostingRegressor(n_estimators=29)
models=[]

models.append(('DT',DT))



models.append(('RF',RF))



models.append(('KNN', KNN))



models.append(('GBoost', GBoost))



import sklearn.model_selection as model_selection

# evaluate each model in turn

results = []

names = []

for name, model in models:

    kfold = model_selection.KFold(shuffle=True, n_splits=7, random_state=0)

    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error' ) # fit, train, predict

    results.append(np.sqrt(np.abs(cv_results)))    # negative mean squared error 

    names.append(name)

    print('%s: %f (%f)'% (name, np.mean(np.sqrt(np.abs(cv_results))), np.var(np.sqrt(np.abs(cv_results)),ddof=1)))

    

    # boxplot algorithm comparision

    fig = plt.figure()

    fig.suptitle('Algorithm Comparision')

    ax = fig.add_subplot(111)

    plt.boxplot(results)

    ax.set_xticklabels(names)

    plt.show()
# evaluate each model in turn



results = []

names = []

for name, model in models:

    kfold = model_selection.KFold(shuffle=True, n_splits=7, random_state=0)

    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error' ) # fit, train, predict

    results.append(np.sqrt(np.abs(cv_results)))    # negative mean squared error 

    names.append(name)

    print('%s: %f (%f)'% (name, np.mean(np.sqrt(np.abs(cv_results))), np.var(np.sqrt(np.abs(cv_results)),ddof=1)))

    

    # boxplot algorithm comparision

    fig = plt.figure()

    fig.suptitle('Algorithm Comparision')

    ax = fig.add_subplot(111)

    plt.boxplot(results)

    ax.set_xticklabels(names)

    plt.show()