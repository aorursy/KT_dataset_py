# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import math
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/av-guided-hackathon/train.csv")
test_data = pd.read_csv("/kaggle/input/av-guided-hackathon/test.csv")
sample_submission = pd.read_csv("/kaggle/input/av-guided-hackathon/sample_submission_cxCGjdN.csv")


%matplotlib inline
plt.style.use("seaborn-dark")


import warnings
warnings.simplefilter('ignore')
train_data.head(3)
Id_col, Target_col = 'video_id', 'likes'
print(f'\n Train set contains  {train_data.shape[0]} samples and  {train_data.shape[1]} variables' )
print(f'\n Test set contains  {test_data.shape[0]} samples and  {test_data.shape[1]} variables' )

features = [col for col in train_data.columns if col not in [Id_col, Target_col]]

print(f'\n Data set contains  {len(features)} features' )
# This is the regression problem sowe can plot PDE to find the distribution of the data.( Target destribution)

_ = train_data[Target_col].plot(kind='density',title='Likes Distribution', fontsize = 14, figsize = (8,5))
_ = pd.Series(np.log1p(train_data[Target_col])).plot(kind='density',title = "Likes Distribution", fontsize=14,figsize= (8,6))
#original target

_  = train_data[Target_col].plot(kind='box',figsize=(8,6),vert=False, fontsize=14,title='Likes boxplot')
#log transformed data

_= pd.Series(np.log1p(train_data[Target_col])).plot(kind='box',vert=False, figsize=(8,6), fontsize=14,title='Log likes boxplot')
#Checking independent variables

train_data.info()

Null_values_per_variable = 100*(train_data[features].isnull().sum()/train_data.shape[0])
Null_values_per_variable.sort_values(ascending=False) 
train_data.nunique()
train_data.columns
Category =  ['category_id','country_code']
numerical = ['views','dislikes','comment_count']
datetime= ['publish_date']
text =['title','tags','description']
fig, axes = plt.subplots(3,1, figsize=(10,8))

for i,c in enumerate(numerical):
   _= train_data[[c]].boxplot(vert=False,ax=axes[i])
sns.set(font_scale=1.3)
fig, axes = plt.subplots(2,2, figsize=(10,8))
numerical = ['views','dislikes','comment_count']
axes = [ax for axes_row in axes for ax in axes_row]

for i,c in enumerate(numerical):
    plot = sns.kdeplot(data = train_data[c],ax=axes[i])
plt.tight_layout()
for col in numerical+ [Target_col]:
    train_data[col] = pd.Series(np.log1p(train_data[col]))
#checking the boxl[lot and PDE again for m=numerical data 
fig, axes = plt.subplots(3,1, figsize=(10,8))

for i,c in enumerate(numerical):
   _= train_data[[c]].boxplot(vert=False,ax=axes[i])
#checking PDE for transformed data
sns.set(font_scale=1.3)
fig, axes = plt.subplots(2,2, figsize=(10,8))
numerical = ['views','dislikes','comment_count']
axes = [ax for axes_row in axes for ax in axes_row]

for i,c in enumerate(numerical):
    plot = sns.kdeplot(data = train_data[c],ax=axes[i])
plt.tight_layout()
plt.figure(figsize = (10,8))
_ = sns.heatmap(train_data[numerical+['likes']].corr(), annot=True)

_ = sns.pairplot(train_data[numerical+['likes']],height=5,aspect=24/16 )
figure, axes = plt.subplots(1,2,figsize=(15,10))


for i,c in enumerate(['category_id','country_code']):
    _ = train_data[c].value_counts()[::-1].plot(kind='pie',title=c,ax=axes[i],fontsize=18,autopct='%.0f')
    _ = axes[i].set_ylabel("")
_ = plt.tight_layout()
#Top 20 channles with highest number of videos
top_20_channels = train_data['channel_title'].value_counts()[:20].reset_index()
top_20_channels.columns = ['channel_title','num_videos']
plt.figure(figsize=(15,10))
_ = sns.barplot(data= top_20_channels,y ='channel_title',x='num_videos')
_ = plt.title("Top 20 channels with highest number of videos")

countrywise = train_data.groupby(['country_code','channel_title']).size().reset_index()
countrywise.columns = ['country_code','channel_title','num_videos']
countrywise = countrywise.sort_values(by='num_videos',ascending=False)
fig,axes= plt.subplots(4,1,figsize=(10,20))

for i, c in enumerate(train_data['country_code'].unique()):
    country = countrywise[countrywise['country_code']==c][:10]
    _ =sns.barplot(data=country,x='num_videos',y='channel_title',ax=axes[i])
    _ = axes[i].set_title(f"Country code {c}")

plt.tight_layout()

countrywise_likes = train_data.groupby(['country_code','channel_title'])['likes'].max().reset_index()
countrywise_likes = countrywise_likes.sort_values(by=['likes'],ascending=False)
fig,axes= plt.subplots(4,1,figsize=(10,20))

for i, c in enumerate(train_data['country_code'].unique()):
    country = countrywise_likes[countrywise_likes['country_code']==c][:10]
    _ =sns.barplot(data=country,x='likes',y='channel_title',ax=axes[i])
    _ = axes[i].set_title(f"Country code {c}")

plt.tight_layout()
sns.catplot(data=train_data,x='category_id',y='likes',height =5, aspect=24/8)
sns.catplot(data=train_data,x='country_code',y='likes',height =5, aspect=24/8)
#countrywise distribution of likes;

_=train_data.groupby(['country_code'])['likes'].mean().sort_values().plot(kind='barh')
train_data['publish_date'] = train_data['publish_date'].astype('datetime64')
train_data['publish_date'].max(), train_data['publish_date'].min()
test_data['publish_date'] = test_data['publish_date'].astype('datetime64')

train_data['publish_date'].dt.year.value_counts()
latest_train = train_data[train_data['publish_date']>'2017-11']
latest_test = test_data[test_data['publish_date']>'2017-11']

_ = latest_train.sort_values(by='publish_date').groupby('publish_date').size().rename('train').plot(figsize=(14,8),title="Number of videos ")
_ = latest_test.sort_values(by='publish_date').groupby('publish_date').size().rename('test').plot(figsize=(14,8),title="Number of videos")
_ = plt.legend()
#Number of likes sorted by data
latest_train = train_data[train_data['publish_date']>'2017-11']
_ = latest_train.sort_values(by='publish_date').groupby('publish_date')['likes'].mean().plot(figsize=(14,8),title="Number of videos ")


country = latest_train.groupby(['country_code','publish_date']).size().reset_index()
_ = country.pivot_table(index='publish_date',columns='country_code',values=0).plot(subplots=True,figsize=(18,18),
                                                                                 title = 'Number of videos countrywise',
                                                                                 sharex=False,
                                                                                 fontsize=18)

#NO. OF LIKES COUNTRYWISE.

country = latest_train.groupby(['country_code','publish_date'])['likes'].mean().reset_index()
_ = country.pivot_table(index='publish_date',columns='country_code',values='likes').plot(subplots=True,figsize=(18,18),
                                                                                 title = 'Number of videos countrywise',
                                                                                 sharex=False,
                                                                                 fontsize=18)
#Do people post more videos on weekdays or weekends?
train_data['dayofweek'] = train_data['publish_date'].dt.dayofweek
videos_per_Day_of_week = train_data['dayofweek'].value_counts().sort_index().reset_index()
videos_per_Day_of_week.columns = ['dayofweek','num_videos']
videos_per_Day_of_week['dayofweek'] = ['Mon','Tue', 'Wed', 'Thru','Fri', 'Sat', 'Sun']
_ = sns.catplot(data=videos_per_Day_of_week,x='dayofweek',y='num_videos',kind='point',aspect=24/6)
_=  plt.title("Number of videos posted per days of week",fontsize=14)
from wordcloud import WordCloud,STOPWORDS
text =['title','tags','description']
wc = WordCloud(stopwords = set(list(STOPWORDS)+['|']),random_state=42)
train_data['likes'].describe()
100* ((train_data['likes']>10).sum())/train_data.shape[0]
def plot_countrywise(country_code='IN'):
    country = train_data[train_data['country_code']==country_code]
    country =country[country['likes']>10]
    fig, axes = plt.subplots(2,2, figsize = (20,12))
    axes = [ax for row_axes in axes for ax in row_axes]
    
    for i,c in enumerate(text):
        op =  wc.generate(str(country[c]))
        _ = axes[i].imshow(op)
        _ = axes[i].set_title(c.upper(),fontsize=20)
        _ = axes[i].axis('off')
        
    fig.delaxes(axes[3])
    plt.suptitle(f"Country code : '{country_code}'",fontsize=30)
    
plot_countrywise('IN')
plot_countrywise('US')
plot_countrywise('GB')
plot_countrywise('CA')
train_data['description_len'] = train_data['description'].apply(lambda x:len(x))
train_data['title_len'] = train_data['title'].apply(lambda x:len(x))
train_data['tags_len'] = train_data['tags'].apply(lambda x:len(x))
train_data['channel_len'] = train_data['channel_title'].apply(lambda x:len(x))

_ = sns.heatmap(train_data[['description_len','title_len','channel_len','tags_len','likes']].corr(),annot=True)
#data load
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor
import pickle
from sklearn.model_selection import train_test_split


train = pd.read_csv("/kaggle/input/av-guided-hackathon/train.csv")
test = pd.read_csv("/kaggle/input/av-guided-hackathon/test.csv")

cat_cols =  ['category_id','country_code','channel_title']
numerical = ['views','dislikes','comment_count']
datetime= ['publish_date']
text =['title','tags','description']
main_features = list(test.columns)
#metric selection


def rmlse(pred,orig):
    return np.sqrt(mean_squared_log_error(orig,pred))

def av_metric(pred,orig):
    return 1000*np.sqrt(mean_squared_error(orig,pred))
    
    
def download_data(pred_test,filename):
    pred_test  = pd.Series(np.expm1(pred_test))
    pred_test = pred_test.round()
    improved = pd.concat([test['video_id'],pred_test],axis=1,).reset_index(drop=True)
    improved.set_index('video_id',inplace=True)
    improved.rename(columns={0:'likes'},inplace=True)
    improved.to_csv(filename)
    

def join_df(train,test):
    df = pd.concat([train,test],axis=0).reset_index(drop=True)
    df[cat_cols] = df[cat_cols].apply(lambda x: pd.factorize(x)[0])    
    features  = [c for c in df.columns if c not in ['likes','video_id']]
    df[numerical+['likes']] = df[numerical+['likes']].apply(lambda x: np.log1p(x))
    
    return df,features

def get_split_features(df,train_nrows):
    
    train_proc,test_proc = df[:train_nrows],df[train_nrows:].reset_index(drop=True)
    cols = [i for i in train_proc.columns if i not in ['likes','video_id']]
    return train_proc,test_proc, cols




def feature_engineering(df):
    ## datetime columns

    df['publish_date'] = pd.to_datetime(df['publish_date'],format ="%Y-%m-%d")

    df['publish_date_days_since_start'] = (df['publish_date']-df['publish_date'].min()).dt.days
    df['publish_date_days_of_weeks'] = df['publish_date'].dt.dayofweek
    df['publish_date_year'] = df['publish_date'].dt.year
    df['publish_date_month'] = df['publish_date'].dt.month
    
    #channel title
    df['channel_title_num_videos']= df['channel_title'].map(df['channel_title'].value_counts())
    df['publish_date_num_videos'] = df['publish_date'].map(df['publish_date'].value_counts())
    #creating more colum
    df['Channel_in_n_countries']=df.groupby(['channel_title'])['country_code'].transform('nunique')
    
    # Grouping features

    df['mean_views_in_channel'] = df.groupby(['channel_title'])['views'].transform('mean')
    df['max_views_in_channel'] = df.groupby(['channel_title'])['views'].transform('max')
    df['min_views_in_channel'] = df.groupby(['channel_title'])['views'].transform('min')


    df['mean_comments_in_channel'] =df.groupby(['channel_title'])['comment_count'].transform('mean')
    df['max_comments_in_channel'] = df.groupby(['channel_title'])['comment_count'].transform('max')
    df['min_comments_in_channel'] = df.groupby(['channel_title'])['comment_count'].transform('min')


    df['mean_dislikes_in_channel'] =df.groupby(['channel_title'])['dislikes'].transform('mean')
    df['max_dislikes_in_channel'] = df.groupby(['channel_title'])['dislikes'].transform('max')
    df['min_dislikes_in_channel'] = df.groupby(['channel_title'])['dislikes'].transform('min')
    
    # Length of text columns

    df['len_of_title_columns'] = df['title'].apply(lambda x:len(x))
    df['len_of_tags_columns'] = df['tags'].apply(lambda x:len(x))
    df['len_of_description_columns'] =df['description'].apply(lambda x:len(x))
    
    #text
    TOP_N_WORDS=50

    #for description
    vect = CountVectorizer(max_features=TOP_N_WORDS)
    txt_to_fts_desc = vect.fit_transform(df['description']).toarray()

    c = 'description'
    txt_fts_names = [c + f"_word_{i}_count" for i in range(TOP_N_WORDS)]
    descrip = pd.DataFrame(txt_to_fts_desc,columns=txt_fts_names)

    #for tags
    vect1 = CountVectorizer(max_features=TOP_N_WORDS)
    txt_to_fts_tags = vect1.fit_transform(df['tags']).toarray()

    c = 'tags'
    txt_fts_names = [c + f"_word_{i}_count" for i in range(TOP_N_WORDS)]
    tags = pd.DataFrame(txt_to_fts_tags,columns=txt_fts_names)

    #for titles
    vect2 = CountVectorizer(max_features=TOP_N_WORDS)
    txt_to_fts_titles = vect2.fit_transform(df['title']).toarray()

    c = 'title'
    txt_fts_names = [c + f"_word_{i}_count" for i in range(TOP_N_WORDS)]
    title = pd.DataFrame(txt_to_fts_titles,columns=txt_fts_names)
    
    #merge
    

    df = pd.concat([df,title,tags,descrip],axis=1).reset_index(drop=True)
     
    return df


def feature_selection(df):
    features = [c for c in df.columns if c not in ['likes','video_id']]
    cat_num_columns = [c for c in features if c not in ['title','tags','description','publish_date']]

    train_proc, test_proc, features = get_split_features(df,train.shape[0])
    

    model = ExtraTreesRegressor()
    model.fit(train_proc[cat_num_columns],train['likes'])
    imp = pd.Series(model.feature_importances_,index=cat_num_columns)
    imp_features  =imp.nlargest(int(0.80*len(cat_num_columns))).index.tolist()
    
    return train_proc,test_proc,imp_features
#function for gradient boosting:
def run_gradient_boosting(clf, fit_params, train, test, features):

    N_splits = 5
    oofs = np.zeros(len(train))
    preds = np.zeros((len(test)))
    
    target = train['likes']
    fold = StratifiedKFold(n_splits= N_splits)
    stratified_target = pd.qcut(train['likes'],10,labels=False,duplicates='drop')
    
    feature_importance= pd.DataFrame()

    for fold_, (trn_idx, val_idx) in enumerate(fold.split(train, stratified_target)):
        print(f'\n------------- Fold {fold_ + 1} -------------')

        ### Training Set
        X_trn, y_trn = train[features].iloc[trn_idx], target.iloc[trn_idx]

        ### Validation Set
        X_val, y_val = train[features].iloc[val_idx], target.iloc[val_idx]

        ### Test Set
        X_test = test[features]

        scaler = StandardScaler()
        _ = scaler.fit(X_trn)

        X_trn = scaler.transform(X_trn)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        _ = clf.fit(X_trn, y_trn, eval_set = [(X_val, y_val)], **fit_params)
        
        fold_importance = pd.DataFrame({'fold':fold_+1,'features':features,'importance':clf.feature_importances_})
        feature_importance = pd.concat([feature_importance,fold_importance],axis=0)
        ### Instead of directly predicting the classes we will obtain the probability of positive class.
        preds_val = clf.predict(X_val)
        preds_test = clf.predict(X_test)

        fold_score = av_metric(preds_val, y_val)
        print(f'\nAV score for validation set is {fold_score}')

        oofs[val_idx] = preds_val
        preds += preds_test / N_splits


    oofs_score = av_metric(oofs,target)
    print(f'\n\n AV score for oofs is {oofs_score}')
    
    
    feature_importance = feature_importance.reset_index(drop=True)
    fi = feature_importance.groupby('features')['importance'].mean().sort_values(ascending=False)[:20][::-1]
    #fi.plot(kind='barh',figsize=(12,6))
    
    return oofs, preds,fi,clf


#preprocessing
def preprocessing(train,test):
    
    df,features = join_df(train,test)
    new_df = feature_engineering(df)
    train_proc,test_proc,imp_features = feature_selection(new_df)
    return train_proc,test_proc,imp_features
  
#catboost final  model

def model(train,test):
    
    train_proc,test_proc,imp_features = preprocessing(train,test)
    
    clf = CatBoostRegressor(learning_rate=0.01,iterations=3000,random_state=2054,task_type='GPU')
    params = {'verbose':False,'early_stopping_rounds':200}
    cat_oofs, cat_preds, fi,model = run_gradient_boosting(clf, params, train_proc, test_proc, imp_features)
    
    download_data(cat_preds,"final_catboost_fe_fs.csv")
    
    return model


YLP_model = model(train,test)

