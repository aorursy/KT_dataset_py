import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import GridSearchCV,StratifiedShuffleSplit,StratifiedKFold,RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC,SVR
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import auc,roc_auc_score, accuracy_score, confusion_matrix, f1_score, precision_score, \
recall_score, matthews_corrcoef, precision_recall_curve
import xgboost as xgb
from xgboost import XGBClassifier,XGBRegressor
from sklearn.preprocessing import PolynomialFeatures,LabelEncoder
from sklearn.feature_selection import VarianceThreshold
pd.set_option('display.max_columns', 100)
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
import lightgbm as lgb
import eli5
import shap

#Credits:
#a lot of the tasks done here is taken from below.
#https://www.kaggle.com/artgor/eda-feature-engineering-and-model-interpretation
from wordcloud import WordCloud
import ast
from collections import Counter
import plotly.graph_objs as go
import plotly.offline as py
def text_to_dic(df,columns):
    for column in columns:
        df[column]=df[column].apply(lambda x:{} if pd.isnull(x) else ast.literal_eval(x))
    return df
train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')
test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')
train_original=train.copy()
test_original=test.copy()
dict_columns = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']
train.head(1) # there are a lot of columns that have srings ( and dictionaries within them). Wil have to think how to fix them
train['belongs_to_collection'][0] # this is a string
def text_to_dic(df,columns):
    for column in columns:
        df[column]=df[column].apply(lambda x:{} if pd.isnull(x) else ast.literal_eval(x))
    return df
train=text_to_dic(train,dict_columns)
test=text_to_dic(test,dict_columns)
for i, value in enumerate(train[dict_columns[0]][0:4]):
    print(i,value)
train['collection_name']=train['belongs_to_collection'].apply(lambda x:x[0]['name'] if x!={} else 0)
train['has_collection_name']=train['belongs_to_collection'].apply(lambda x:0 if x=={} else 1)
#delete the original column
train.drop(['belongs_to_collection','collection_name'],axis=1,inplace=True)
test['collection_name']=test['belongs_to_collection'].apply(lambda x:x[0]['name'] if x!={} else 0)
test['has_collection_name']=test['belongs_to_collection'].apply(lambda x:0 if x=={} else 1)
#delete the original column
test.drop(['belongs_to_collection','collection_name'],axis=1,inplace=True)
train['genres']=train['genres'].apply(lambda x: [i['name'] for i in x] if x!={} else [])
train['genre_count']=train['genres'].apply(lambda x: len(x) if x!={} else 0)
#test
test['genres']=test['genres'].apply(lambda x: [i['name'] for i in x] if x!={} else [])
test['genre_count']=test['genres'].apply(lambda x: len(x) if x!={} else 0)
list_of_genres=[i for i in train['genres']]
plt.figure(figsize = (10, 6))
text = ' '.join([i for j in list_of_genres for i in j])
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top genres')
plt.axis("off")
plt.show()
list_of_genres_test=[i for i in test['genres']]
plt.figure(figsize = (10, 6))
text = ' '.join([i for j in list_of_genres_test for i in j])
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top genres')
plt.axis("off")
plt.show()
all_genres=set()
for i in list_of_genres:
    all_genres=all_genres.union(i)
all_genres_test=set()
for i in list_of_genres_test:
    all_genres_test=all_genres_test.union(i)
print(all_genres_test),print(len(all_genres_test))
most_common_genre=Counter([i for j in list_of_genres for i in j]).most_common(15)
for genre in most_common_genre:
    train['genre_'+genre[0]]=train['genres'].apply(lambda x: 1 if genre[0] in x else 0)
    test['genre_'+genre[0]]=test['genres'].apply(lambda x: 1 if genre[0] in x else 0)
    
#drop genre colum

train.drop('genres',axis=1,inplace=True)
test.drop('genres',axis=1,inplace=True)
for i in train['production_companies'][0:4]:
    print(i)
train['production_cos']=train['production_companies'].apply(lambda x :[i['name']  for i in x] if x!={} else [] )
test['production_cos']=test['production_companies'].apply(lambda x :[i['name']  for i in x] if x!={} else [] )
#take the count
train['Count_production_cos']=train['production_cos'].apply( lambda x: 0 if x==[] else len(x) )
test['Count_production_cos']=test['production_cos'].apply( lambda x: 0 if x==[] else len(x) )
#lets create binary columns for th top 10 most occuring production firms
most_common_production_cos=Counter([i for j in train['production_cos'] for i in j]).most_common(10)
for i in most_common_production_cos:
    train['Prod_cos_'+i[0]]=train['production_cos'].apply(lambda x: 1 if i[0] in x else 0)
    test['Prod_cos_'+i[0]]=test['production_cos'].apply(lambda x: 1 if i[0] in x else 0)
train.drop(['production_companies','production_cos'],axis=1,inplace=True)
test.drop(['production_companies','production_cos'],axis=1,inplace=True)
for i in train['production_countries'][18:20]:
    print(i)
train['prod_countries']=train['production_countries'].apply(lambda x:[i['name'] for i in x] if x!={} else [])
test['prod_countries']=test['production_countries'].apply(lambda x:[i['name'] for i in x] if x!={} else [])
train['prod_country_count']=train['production_countries'].apply(lambda x:0 if x=={} else len(x))
test['prod_country_count']=test['production_countries'].apply(lambda x:0 if x=={} else len(x))
most_common_production_countries=Counter([i for j in train['prod_countries'] for i in j]).most_common(10) #take top 10 countries
for i in most_common_production_countries:
    train['prod_country_'+i[0]]=train['prod_countries'].apply(lambda x:1 if i[0] in x else 0)
    test['prod_country_'+i[0]]=test['prod_countries'].apply(lambda x:1 if i[0] in x else 0)
train.drop(['production_countries','prod_countries'],axis=1,inplace=True)
test.drop(['production_countries','prod_countries'],axis=1,inplace=True)
train['Language']=train['spoken_languages'].apply(lambda x:[i['name'] for i in x] if x!={} else [])
test['Language']=test['spoken_languages'].apply(lambda x:[i['name'] for i in x] if x!={} else [])
#count
train['Count_Language']=train['Language'].apply(lambda x:len(x) if x!={} else 0)
test['Count_Language']=test['spoken_languages'].apply(lambda x:len(x) if x!={} else 0)
most_common_languages=Counter([i for j in train['Language'] for i in j]).most_common(9) 
for i in most_common_languages:
    train['Language_'+i[0]]=train['Language'].apply(lambda x: 1 if i[0] in x else 0)
    test['Language_'+i[0]]=test['Language'].apply(lambda x: 1 if i[0] in x else 0)
train.drop(['spoken_languages','Language'],axis=1,inplace=True)
test.drop(['spoken_languages','Language'],axis=1,inplace=True)
train['Keywords']=train['Keywords'].apply(lambda x:[i['name'] for i in x] if x!={} else [])
test['Keywords']=test['Keywords'].apply(lambda x:[i['name'] for i in x] if x!={} else [])
#count
train['Count_Keywords']=train['Keywords'].apply(lambda x:len(x) if x!={} else 0)
test['Count_Keywords']=test['Keywords'].apply(lambda x:len(x) if x!={} else 0)
list_of_Keywords_train=[i for i in train['Keywords']]
plt.figure(figsize = (10, 6))
text = ' '.join(['_'.join(i.split(' ')) for j in list_of_Keywords_train for i in j]) 
#Since keywords for the same records may have spaces, added a _ to avoid mixing them with words from other records
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top Keywords')
plt.axis("off")
plt.show()
most_common_Keywords=Counter([i for j in train['Keywords'] for i in j]).most_common(10) 
for i in most_common_Keywords:
    train['Keywords_'+i[0]]=train['Keywords'].apply(lambda x:1 if i[0] in x else 0)
    test['Keywords_'+i[0]]=test['Keywords'].apply(lambda x:1 if i[0] in x else 0)
train.drop('Keywords',axis=1,inplace=True)
test.drop('Keywords',axis=1,inplace=True)
train['Cast_name']=train['cast'].apply(lambda x:[i['name'] for i in x] if x!={} else [])
test['Cast_name']=test['cast'].apply(lambda x:[i['name'] for i in x] if x!={} else [])
#count cast
train['Cast_count']=train['Cast_name'].apply(lambda x:0 if x==[] else len(x))
test['Cast_count']=test['Cast_name'].apply(lambda x:0 if x==[] else len(x))
most_common_Cast_names=Counter([i for j in train['Cast_name'] for i in j]).most_common(15)
for i in most_common_Cast_names:
    train['Cast_names_'+i[0]]=train['Cast_name'].apply(lambda x: 1 if i[0] in x else 0)
    test['Cast_names_'+i[0]]=test['Cast_name'].apply(lambda x: 1 if i[0] in x else 0)
#gender in cast
# gender of all the cast is not important. I wanted to check if the actor in lead role is male or female But there is no
#way to get this info. We can however have a count of #of male, #of Female and #of Unasigned gender as 3 column in the data
train['Cast_gender']=train['cast'].apply(lambda x:[i['gender'] for i in x]if x !=[] else [0]) #male 2,Female 1 & 0 is Unassigned
test['Cast_gender']=test['cast'].apply(lambda x:[i['gender'] for i in x] if x !=[] else [0])
train['Cast_male_2']=train['Cast_gender'].apply(lambda x:sum([i==2 for i in x]) if x!=[] else 0)
train['Cast_female_1']=train['Cast_gender'].apply(lambda x:sum([i==1 for i in x]) if x!=[] else 0)
train['Cast_gender_UnAss0']=train['Cast_gender'].apply(lambda x:sum([i==0 for i in x]) if x!=[] else 0)

#for test

test['Cast_male_2']=test['Cast_gender'].apply(lambda x:sum([i==2 for i in x]) if x!=[] else 0)
test['Cast_female_1']=test['Cast_gender'].apply(lambda x:sum([i==1 for i in x]) if x!=[] else 0)
test['Cast_gender_UnAss0']=test['Cast_gender'].apply(lambda x:sum([i==0 for i in x]) if x!=[] else 0)
train.drop(['cast','Cast_name','Cast_gender'],axis=1,inplace=True)
test.drop(['cast','Cast_name','Cast_gender'],axis=1,inplace=True)
train['crew_gender']=train['crew'].apply(lambda x:[i['gender'] for i in x] if x!={} else [])
test['crew_gender']=test['crew'].apply(lambda x:[i['gender'] for i in x] if x!={} else [])
train['crew_gender_0']=train['crew_gender'].apply(lambda x:sum([i==0 for i in x]) if x!=[] else 0)
train['crew_gender_1']=train['crew_gender'].apply(lambda x:sum([i==1 for i in x]) if x!=[] else 0)
train['crew_gender_2']=train['crew_gender'].apply(lambda x:sum([i==2 for i in x]) if x!=[] else 0)
test['crew_gender_0']=test['crew_gender'].apply(lambda x:sum([i==0 for i in x]) if x!=[] else 0)
test['crew_gender_1']=test['crew_gender'].apply(lambda x:sum([i==1 for i in x]) if x!=[] else 0)
test['crew_gender_2']=test['crew_gender'].apply(lambda x:sum([i==2 for i in x]) if x!=[] else 0)
train.drop(['crew','crew_gender'],axis=1,inplace=True)
test.drop(['crew','crew_gender'],axis=1,inplace=True)
train['log_budget'] = np.log1p(train['budget'])
test['log_budget'] = np.log1p(test['budget'])
train['runtime'].fillna(value=train['runtime'].mean(),inplace=True)
test['runtime'].fillna(value=train['runtime'].mean(),inplace=True)
#since there are nulls, lets create a binary to mark 1 /0 corrosponding to if Homepage is present or not
train['has_homepage'] = 1
train.loc[train['homepage'].isnull(),'has_homepage']=0
test['has_homepage'] = 1
test.loc[test['homepage'].isnull(),'has_homepage']=0
#Original title ahs the title of the movie. This may not make sense for non english movies and i doubt if this will be 
#useful for the Model. However lets just take a look at the word cloud
plt.figure(figsize = (10, 6))
text = ' '.join(train['original_title'].values)
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top genres')
plt.axis("off")
plt.show()
test.loc[test['release_date'].isnull(),'release_date']='01/01/98'
def fix_date(x):
    """
    fix dates
    """
    year=x.split('/')[2]
    if int(year)<20:
        return x[:-2]+'20'+year
    else:
        return x[:-2]+'19'+year
    
train['release_date']=train['release_date'].apply(lambda x:fix_date(x))  
test['release_date']=test['release_date'].apply(lambda x:fix_date(x))
train['release_date'] = pd.to_datetime(train['release_date'])
test['release_date'] = pd.to_datetime(test['release_date'])
def process_date(df):
    date_parts = ["year", "weekday", "month", 'weekofyear', 'day', 'quarter']
    for part in date_parts:
        part_col = 'release_date' + "_" + part
        df[part_col] = getattr(df['release_date'].dt, part).astype(int)
    
    return df

train = process_date(train)
test = process_date(test)
def new_features(df):
    df['budget_to_popularity'] = df['budget'] / df['popularity']
    df['budget_to_runtime'] = df['budget'] / df['runtime']     #runtime has some 0's so wil result in INF
    
    # some features from https://www.kaggle.com/somang1418/happy-valentines-day-and-keep-kaggling-3
    df['_budget_year_ratio'] = df['budget'] / (df['release_date_year'] * df['release_date_year'])
    df['_releaseYear_popularity_ratio'] = df['release_date_year'] / df['popularity']
    df['_releaseYear_popularity_ratio2'] = df['popularity'] / df['release_date_year']
    #df.groupby("release_date_year")["runtime"].transform('mean') 
    #this gives the value of the avg corresponding to the release_date_year
    df['runtime_to_mean_year'] = df['runtime'] / df.groupby("release_date_year")["runtime"].transform('mean') 
    #runtime has some 0's so wil result in INF
    df['popularity_to_mean_year'] = df['popularity'] / df.groupby("release_date_year")["popularity"].transform('mean')
    df['budget_to_mean_year'] = df['budget'] / df.groupby("release_date_year")["budget"].transform('mean')
        
    return df
train[train['budget']==0].shape[0]/train.shape[0] #27% in train has Budget=0
test[test['budget']==0].shape[0]/test.shape[0]    #27% in test has Budget=0
#before running the above feature engineering steps, we have to fix budget=0 as this is definitely not correct.
train['popularity'].describe()
sum(train['popularity']<1)/train.shape[0]
sum(test['popularity']<1)/test.shape[0] #7-8% in train/test has popularity<1,max value is 250
#so fix popularity before creating the feature enginnered variables.
sns.distplot(train['popularity']);
#lets replace the popularity by its Z values
from sklearn import preprocessing
scaler=preprocessing.StandardScaler()
train['popularity']=5+scaler.fit_transform(np.array(train['popularity']).reshape(-1,1))#added 5 to avoid the values being 0 (z can have 0 as a value)

#replace runtime ==0 with avg of runtime in the release year
train.loc[(train['runtime']==0),'runtime']=train.groupby(['release_date_year'])['runtime'].transform('mean')[train[(train['runtime']==0)].index]
test.loc[(test['runtime']==0),'runtime']=test.groupby(['release_date_year'])['runtime'].transform('mean')[test[(test['runtime']==0)].index]

#replace budget==0 with avg budget of the year and the original_language.
train.loc[(train['budget']==0),'budget']=\
            train.groupby(['release_date_year','original_language'])['budget'].transform('mean')[train[(train['budget']==0)].index]
test.loc[(test['budget']==0),'budget']=\
            test.groupby(['release_date_year','original_language'])['budget'].transform('mean')[test[(test['budget']==0)].index]
#if there are still budget==0, fill them with avg of just the language
train.loc[(train['budget']==0),'budget']=\
            train.groupby(['original_language'])['budget'].transform('mean')[train[(train['budget']==0)].index]
test.loc[(test['budget']==0),'budget']=\
            test.groupby(['original_language'])['budget'].transform('mean')[test[(test['budget']==0)].index]

#and if still use just the release year t0 fill
train.loc[(train['budget']==0),'budget']=\
            train.groupby(['release_date_year'])['budget'].transform('mean')[train[(train['budget']==0)].index]
test.loc[(test['budget']==0),'budget']=\
            test.groupby(['release_date_year'])['budget'].transform('mean')[test[(test['budget']==0)].index]
train=new_features(train)
test=new_features(test)
#Surprisingly films releases on Wednesdays and on Thursdays tend to have a higher revenue.
train.drop('release_date',axis=1,inplace=True)
test.drop('release_date',axis=1,inplace=True)
###Text Columns. From columns that are pure texts lets take the len of each col as a feature
for col in ['title', 'tagline', 'overview', 'original_title']:
    train['len_' + col] = train[col].fillna('').apply(lambda x: len(str(x)))
    train['words_' + col] = train[col].fillna('').apply(lambda x: len(str(x.split(' '))))
    #train = train.drop(col, axis=1)
    test['len_' + col] = test[col].fillna('').apply(lambda x: len(str(x)))
    test['words_' + col] = test[col].fillna('').apply(lambda x: len(str(x.split(' '))))
    #test = test.drop(col, axis=1)
train = train.drop(['homepage', 'imdb_id','original_title','overview','poster_path',\
                    'status','tagline', 'title'], axis=1)
test = test.drop(['homepage', 'imdb_id','original_title','overview','poster_path',\
                    'status','tagline', 'title'], axis=1)

#Check is all values are same in any column
for col in train.columns:
    if train[col].nunique() == 1:
        print(col)
        train = train.drop([col], axis=1)
        test = test.drop([col], axis=1)
train.head()
list(train.dtypes[train.dtypes.values=='object'].index)
for i in list(train.dtypes[train.dtypes.values=='object'].index):
    le=LabelEncoder()
    le.fit(list(train[i].fillna(''))+list(test[i].fillna('')))
    train[i] = le.transform(train[i].fillna('').astype(str))
    test[i]  = le.transform(test[i].fillna('').astype(str))
X_train = train.drop(['id', 'revenue'], axis=1)
y_train = train['revenue']
y_train_log=np.log1p(y_train)
X_test = test.drop(['id'], axis=1)
X_train.shape,X_test.shape
import re
#since columns names have speacial character this is needed
#M a big fan of HINDI Movies-->Language_हिन्दी'
X_train.rename({'Language_日本語':'Language_Japan','Language_普通话':'Language_China','Language_हिन्दी':'Language_India'},axis=1,inplace=True)
X_train.columns=["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_train.columns]
X_train = X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

X_test.rename({'Language_日本語':'Language_Japan','Language_普通话':'Language_China','Language_हिन्दी':'Language_India'},axis=1,inplace=True)
X_test.columns=["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_test.columns]
X_test = X_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

def rmse(y_true, y_pred):
    diff = mean_squared_error(y_true, y_pred)
    return diff**0.5
my_scorer = make_scorer(rmse,greater_is_better=False)
from catboost import CatBoostRegressor
#Provide a K-fold function that generate out-of-fold predictions for train data.
class Modelling():
    def __init__(self,X,y,test_X,folds,N):
        self.X=X
        self.y=y
        self.test_X=test_X
        self.folds=folds
        self.N=N
     
    def Single_Model(self,Regressor): #for all other Models like LInear,NB ,KNN etc
        stacker_train = np.zeros((self.X.shape[0], 1))
        stacker_test  = np.zeros(self.test_X.shape[0])        
        for index, (trn_idx,val_idx) in enumerate(self.folds.split(self.X)):
            trn_x, val_x = self.X[trn_idx], self.X[val_idx]
            trn_y, val_y = self.y[trn_idx], self.y[val_idx]
            print('Train model in fold {}'.format(index+1))           
            Regressor.fit(trn_x,np.log1p(trn_y))
            val_pred = np.expm1(Regressor.predict(val_x))
            stacker_train[val_idx,0]=val_pred
            val_rmse=rmse(val_y, val_pred)            
            print('fold {} RMSE score on VAL is {:.6f}'.format(index+1, val_rmse))
            #for test
            pred_test= np.expm1(Regressor.predict(self.test_X))
            stacker_test+=(pred_test/self.N)
            
        #evaluate for entire train data (oof)
        train_rmse=rmse(self.y,stacker_train)
        print("CV score on TRAIN (OOF) is RMSE: {}".format(train_rmse))   
        return stacker_test,stacker_train        
        
        
        
    def SingleRF_oof(self,params):
        clf_rf=RandomForestRegressor(**rf_params)
        stacker_train = np.zeros((self.X.shape[0], 1))
        stacker_test  = np.zeros(self.test_X.shape[0])
        for index, (trn_idx,val_idx) in enumerate(self.folds.split(self.X,self.y)):
            trn_x, val_x = self.X[trn_idx], self.X[val_idx]
            trn_y, val_y = self.y[trn_idx], self.y[val_idx]
            print('Train model in fold {}'.format(index+1))         
            clf_rf.fit(trn_x,trn_y)
            val_pred = clf_rf.predict(val_x)
            stacker_train[val_idx,0]=val_pred
            val_rmse=rmse(val_y, val_pred)    
                        
            print('fold {} RMSE score on VAL is {:.6f}'.format(index+1,val_rmse))
            #for test
            pred_test= clf_rf.predict(self.test_X)
            stacker_test+=(pred_test/self.N)
        #evaluate for entire train data (oof)
        train_rmse=rmse(self.y,stacker_train)
        print("CV score on TRAIN (OOF) is RMSE: {}".format(train_rmse))   
        return stacker_test,stacker_train    

    
    def SingleXGB_oof(self,params,num_boost_round):
        stacker_train = np.zeros((self.X.shape[0], 1))
        stacker_test=np.zeros(self.test_X.shape[0])
        dtest=xgb.DMatrix(self.test_X)
        for index, (trn_idx,val_idx) in enumerate(self.folds.split(self.X)):
            trn_x, val_x = self.X[trn_idx], self.X[val_idx]
            trn_y, val_y = self.y[trn_idx], self.y[val_idx]
            dtrn = xgb.DMatrix(data=trn_x, label=np.log1p(trn_y))
            dval = xgb.DMatrix(data=val_x, label=np.log1p(val_y))
            print('Train model in fold {}'.format(index+1)) 
            cv_model = xgb.train(params=params,dtrain=dtrn,num_boost_round=num_boost_round\
                                 ,evals=[(dtrn, 'train'), (dval, 'val')],verbose_eval=10,early_stopping_rounds=200)
                        
            pred_test = np.expm1(cv_model.predict(dtest, ntree_limit=cv_model.best_ntree_limit))
            stacker_test+=(pred_test/self.N)
            val_pred=np.expm1(cv_model.predict(dval, ntree_limit=cv_model.best_ntree_limit))
            stacker_train[val_idx,0]=val_pred
            val_rmse=rmse(val_y, val_pred)
            
            print('fold {} RMSE score on VAL is {:.6f}'.format(index+1, val_rmse))
            
        #evaluate for entire train data (oof)
        train_rmse=rmse(self.y,stacker_train)
        print("CV score on TRAIN (OOF) is RMSE: {}".format(train_rmse))   
        return stacker_test,stacker_train
    
    
    def SingleLGBM_oof(self,params,num_boost_round,colnames,importance_plot=False): #passing the col names to print the Feature imp
        stacker_train = np.zeros((self.X.shape[0], 1))
        stacker_test=np.zeros(self.test_X.shape[0])
        feature_importance =pd.DataFrame()
        for index, (trn_idx,val_idx) in enumerate(self.folds.split(self.X,self.y)):
            trn_x, val_x = self.X[trn_idx], self.X[val_idx]
            trn_y, val_y = self.y[trn_idx], self.y[val_idx]

            print('Train model in fold {}'.format(index+1)) 
            lgb_train = lgb.Dataset(trn_x,np.log1p(trn_y))
            lgb_val = lgb.Dataset(val_x, np.log1p(val_y), reference=lgb_train)
            
            lgb_model = lgb.train(params,
                        lgb_train,
                        num_boost_round=num_boost_round,
                        valid_sets=lgb_val,
                        early_stopping_rounds=200,
                        verbose_eval=10)
            
            val_pred=np.expm1(lgb_model.predict(val_x))
            val_rmse=rmse(val_y, val_pred)
            print('fold {} RMSE score on VAL is {:.6f}'.format(index+1, val_rmse))
            stacker_train[val_idx,0]=val_pred

            pred_test = np.expm1(lgb_model.predict(self.test_X))
            stacker_test+=(pred_test/self.N)
            #feature importance
            fold_importance = pd.DataFrame()
            
            fold_importance["feature"] = colnames
            fold_importance["importance"] = lgb_model.feature_importance()
            fold_importance["fold"] = index+1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
        
        if importance_plot:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:30].index
            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]
            plt.figure(figsize=(12, 9));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGBM Features (avg over folds,Top Few)');
                
        
        #evaluate for entire train data (oof)
        train_rmse=rmse(self.y,stacker_train)
        print("CV score on TRAIN (OOF) is RMSE: {}".format(train_rmse))   
        return stacker_test,stacker_train
    
    
    def SingleCatBoost_oof(self,params): #simple catboost without the cat columns
        stacker_train = np.zeros((self.X.shape[0], 1))
        stacker_test=np.zeros(self.test_X.shape[0])
        
        for index, (trn_idx,val_idx) in enumerate(self.folds.split(self.X)):
            trn_x, val_x = self.X[trn_idx], self.X[val_idx]
            trn_y, val_y = self.y[trn_idx], self.y[val_idx]
            print('Train model in fold {}'.format(index+1))              
                
            cat_model = CatBoostRegressor(**params)
            cat_model.fit(trn_x,np.log1p(trn_y),eval_set=(val_x,np.log1p(val_y)),use_best_model=True,verbose=False)
            val_pred = np.expm1(cat_model.predict(val_x))
            stacker_train[val_idx,0]=val_pred
            val_rmse=rmse(val_y, val_pred)            
            print('fold {} RMSE score on VAL is {:.6f}'.format(index+1, val_rmse))
            #for test
            pred_test= np.expm1(cat_model.predict(self.test_X))
            stacker_test+=(pred_test/self.N)
            
        #evaluate for entire train data (oof)
        train_rmse=rmse(self.y,stacker_train)
        print("CV score on TRAIN (OOF) is RMSE: {}".format(train_rmse))   
        return stacker_test,stacker_train   
#call the models
from sklearn.model_selection import KFold
Number_of_folds = 5
#We have to make sure same K fold splits are used for all Models. This avoids Overfitting and Leakage
folds = KFold(n_splits=Number_of_folds, shuffle=True, random_state=2017)
modelling_object = Modelling(X=X_train.values, y=y_train.values, test_X=X_test.values, folds=folds, N=Number_of_folds)
#NOTE above that we are calling train, test all as ..values
#Call LightGBM
cat_params= {
    'iterations':10000,
    'learning_rate':0.004,
   'depth':5,
    'eval_metric':'RMSE',
    'colsample_bylevel':0.8,
    'random_seed' : 2017,
    'bagging_temperature' : 0.2,
    'early_stopping_rounds':200
} 
test_pred_stacked_cat,stacker_train_cat=\
modelling_object.SingleCatBoost_oof(params=cat_params)
#Call LightGBM
lgbm_params= {#"max_depth": 5,          #max depth for tree model
              "learning_rate" : 0.02,
              #"num_leaves": 25,        #max number of leaves in one tree
              # 'feature_fraction':0.6,  #LightGBM will randomly select part of features on each tree node
               'bagging_fraction':0.6,    #randomly select part of data without resampling
              # 'max_drop': 5,         #used only in dart,max number of dropped trees during one boosting iteration
               'lambda_l1': 1,
               'lambda_l2': 0.01,
              'min_child_samples':400,  #minimal number of data in one leaf
                'max_bin':20, #max number of bins that feature values will be bucketed in. Higher value--> Overfitting
                'subsample':0.6,  #randomly select part of data without resampling
                'colsample_bytree':0.8, #same as feature_fraction
               'boosting_type': 'dart',   #options are gbdt(gradientboosting decision trees), rf,dart,goss
               'task': 'train'}  #weight of labels with positive class

test_pred_stacked_lgbm,stacker_train_lgbm=\
modelling_object.SingleLGBM_oof(params=lgbm_params,num_boost_round=10000,colnames=X_train.columns,importance_plot=True)
#Call XGB
params_for_xgb = {
    'objective': 'reg:squarederror',  #the learning task and the corresponding learning objective
    'eval_metric': 'rmse',            #Evaluation metrics for validation data
    'eta': 0.04,          #learning_rate          
    'max_depth': 3,       #Maximum depth of a tree. High will make the model more complex and more likely to overfit.
    'min_child_weight': 5, #[0,inf] Higher the value,lesser the number of splits
    'gamma': 1.5,       #Minimum loss reduction required to make a further partition on a leaf node of the tree    'subsample': 0.8,    #Subsample ratio of the training instances
    'colsample_bytree': 0.6,  #subsample ratio of columns when constructing each tree
    'alpha': 5,  #L1 regularization term on weights
    'lambda': 5,
    'subsample':0.6,
    'seed': 2017}

test_pred_stacked_xgb,stacker_train_xgb=modelling_object.SingleXGB_oof(params=params_for_xgb,num_boost_round=10000) 
rf_params = {'n_estimators': 2000,
              'max_features': 'auto', #, 'sqrt','auto'
              #'criterion':  'gini', #'entropy',
              'max_depth': 30,
              'min_samples_leaf': 15,
            # 'min_samples_split':5,
            # 'class_weight':'balanced',
             'random_state':0
            }

test_pred_stacked_rf,stacker_train_rf=modelling_object.SingleRF_oof(params=rf_params)
#Stacking
columns=['catboost','xgb','lgbm','rf']
train_pred_df_list=[stacker_train_cat,stacker_train_xgb, stacker_train_lgbm, stacker_train_rf]
test_pred_df_list=[test_pred_stacked_cat,test_pred_stacked_xgb,test_pred_stacked_lgbm,test_pred_stacked_rf]
lv1_train_df=pd.DataFrame(columns=columns)
lv1_test_df=pd.DataFrame(columns=columns)
for i in range(len(columns)):
    lv1_train_df[columns[i]]=train_pred_df_list[i][:,0]
    lv1_test_df[columns[i]]=test_pred_df_list[i]
    
lv1_train_df['Y']=y_train #add the dependendt variable to training
l2_modelling_object = Modelling(X=lv1_train_df.drop('Y',axis=1).values, y=lv1_train_df['Y'].values, \
                                test_X=lv1_test_df.values, folds=folds, N=5)
test_pred_stacked_lgbm_L2,stacker_train_lgbm_L2=\
l2_modelling_object.SingleLGBM_oof(params=lgbm_params,num_boost_round=10000,colnames=columns,importance_plot=True)
#XGB Model scores 1.98 on LB.( Best out of all models) If I edit the data 
#like a lot of kernels have done, i.e. correct the revenue/budget in train dataset for many movies, i probably will get better at LB
#Stacking doesnot help and in fact scores poorly
results=pd.DataFrame({'id':test['id'],'revenue':test_pred_stacked_lgbm_L2})
results.to_csv('All_models_stacked_lgbm_L2.csv',index=False)
from IPython.display import FileLinks
FileLinks('.')








results=pd.DataFrame({'id':test['id'],'revenue':test_pred_stacked_lm_L2})
results.to_csv('Linear_Model_L2.csv',index=False)



from IPython.display import FileLinks
FileLinks('.')







