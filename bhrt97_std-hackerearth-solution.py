import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# NLP
from bs4 import BeautifulSoup
import string
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
df= pd.read_csv("/kaggle/input/hackerearth-effectiveness-of-std-drugs/dataset/train.csv")
to_test = pd.read_csv("/kaggle/input/hackerearth-effectiveness-of-std-drugs/dataset/test.csv")
df.describe(include='all')
to_test.head(5)
def punctuation_removal(messy_str):
    clean_list = [char for char in messy_str if char not in string.punctuation]
    clean_str = ''.join(clean_list)
    return clean_str

df['review_by_patient'] = df['review_by_patient'].apply(punctuation_removal)
to_test['review_by_patient'] = to_test['review_by_patient'].apply(punctuation_removal)
stop = stopwords.words('english')
stop.append("i'm")

stop_words = []

for item in stop: 
    new_item = punctuation_removal(item)
    stop_words.append(new_item) 

def stopwords_removal(messy_str):
    messy_str = word_tokenize(messy_str)
    return [word.lower() for word in messy_str 
            if word.lower() not in stop_words ]

df['review_by_patient'] = df['review_by_patient'].apply(stopwords_removal)
to_test['review_by_patient'] = to_test['review_by_patient'].apply(stopwords_removal)
import re

def drop_numbers(list_text):
    list_text_new = []
    for i in list_text:
        if not re.search('\d', i):
            list_text_new.append(i)
    return ' '.join(list_text_new)

df['review_by_patient'] = df['review_by_patient'].apply(drop_numbers)
to_test['review_by_patient'] = to_test['review_by_patient'].apply(drop_numbers)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Create list (cast to array) of compound polarity sentiment scores for reviews
train_sentiments = []
test_sentiments = []

for i in df.review_by_patient:
    train_sentiments.append(sid.polarity_scores(i).get('compound'))
    
for i in to_test.review_by_patient:
    test_sentiments.append(sid.polarity_scores(i).get('compound'))   
    
train_sentiments = np.asarray(train_sentiments)
test_sentiments  = np.asarray(test_sentiments)
df['sentiment'] = pd.Series(data=train_sentiments)
to_test['sentiment'] = pd.Series(data=test_sentiments)
df.sentiment.hist(color='skyblue', bins=30)
plt.title('Compound Sentiment Score Distribution')
plt.xlabel('Scores')
plt.ylabel('Count')
temp_ls = []

for i in range(1, 11):
    temp_ls.append(np.sum(df[df.effectiveness_rating == i].sentiment) / np.sum(df.effectiveness_rating == i))
    

plt.scatter(x=range(1, 11), y=temp_ls, c=range(1, 11), cmap='tab10', s=200)
plt.title('Average Sentiment vs Rating')
plt.xlabel('Rating')
plt.ylabel('Average Sentiment')
plt.xticks([i for i in range(1, 11)])
name_of_drug = df.name_of_drug.value_counts().sort_values(ascending=False)
name_of_drug[:10]
use_case_for_drug = df.use_case_for_drug.value_counts().sort_values(ascending=False)
use_case_for_drug[:10]
# Look at bias in review (also shown on 'Data' page in competition: distribution of ratings)
plt.rcParams['figure.figsize'] = [12, 8]
df.effectiveness_rating.hist(color='skyblue')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.xticks([i for i in range(1, 11)])
# Create a list (cast into an array) containing the average usefulness for given ratings
use_ls = []

for i in range(1, 11):
    use_ls.append([i, np.sum(df[df.effectiveness_rating == i].number_of_times_prescribed) / np.sum([df.effectiveness_rating == i])])
    
use_arr = np.asarray(use_ls)
plt.scatter(use_arr[:, 0], use_arr[:, 1], c=use_arr[:, 0], cmap='tab10', s=200)
plt.title('Average Useful Count vs Rating')
plt.xlabel('Rating')
plt.ylabel('Average Useful Count')
plt.xticks([i for i in range(1, 11)]);plt.scatter(use_arr[:, 0], use_arr[:, 1], c=use_arr[:, 0], cmap='tab10', s=200)
plt.title('Average Useful Count vs Rating')
plt.xlabel('Rating')
plt.ylabel('Average Useful Count')
plt.xticks([i for i in range(1, 11)])
# Sort train dataframe from most to least useful
df = df.sort_values(by='number_of_times_prescribed', ascending=False)
df.iloc[:10]
df= pd.read_csv("/kaggle/input/hackerearth-effectiveness-of-std-drugs/dataset/train.csv")
to_test = to_test[["patient_id","name_of_drug","use_case_for_drug","review_by_patient","effectiveness_rating","drug_approved_by_UIC","number_of_times_prescribed"]]
to_test.head(1)
# ################################


min_rating = df.effectiveness_rating.min()
max_rating = df.effectiveness_rating.max()

def scale_rating(rating):
    rating -= min_rating
    rating = rating/(max_rating -1)
    rating *= 5
    rating = int(round(rating,0))
    
    if(int(rating) == 0 or int(rating)==1 or int(rating)==2):
        return 0
    else:
        return 1
    


    
df['new_eff_score'] = df.effectiveness_rating.apply(scale_rating)
to_test['new_eff_score'] = df.effectiveness_rating.apply(scale_rating)

# X.drop("effectiveness_rating",axis=1,inplace=True)
training_conditions=df['use_case_for_drug'].unique()
testing_conditions=to_test['use_case_for_drug'].unique()
for i in training_conditions:
    if i not in testing_conditions:
        print(i)
big_df = pd.concat([df.drop("base_score",axis=1), to_test], ignore_index=True)
print(df.shape)
print(to_test.shape)
print(big_df.shape)
# Make dictionary of use_case_of_drug, each value will be a dataframe of all of the drugs used to treat the given use_case
help_dict = {}

# Iterate over conditions
for i in big_df.use_case_for_drug.unique():
    
    temp_ls = []
    
    # Iterate over drugs within a given condition
    for j in big_df[big_df.use_case_for_drug == i].name_of_drug.unique():
        
        # If there are at least 0 reviews for a drug, save its name and average rating in temporary list
        if np.sum(big_df.name_of_drug == j) >= 0:
            temp_ls.append((j, np.sum(big_df[big_df.name_of_drug == j].effectiveness_rating) / np.sum(big_df.name_of_drug == j)))
        
    # Save temporary list as a dataframe as a value in help dictionary, sorted best to worst drugs
    help_dict[i] = pd.DataFrame(data=temp_ls, columns=['drug', 'average_rating']).sort_values(by='average_rating', ascending=False).reset_index(drop=True)
df.head(5)
# # Top 10 drugs of Birth Control
help_dict['Birth Control'].iloc[:10]
count = 0
def rank_drug_test(name):
    global count 
    use_case = to_test.iloc[count][2]
    count = count + 1
#     print(use_case)
    a = help_dict[use_case]   ## this will create a dataframe
    rank=a.index[a['drug'] == name].tolist()
    
    if(rank[0] > 30 ):                                  ## those who has less than 30 reviews are ranked zero
        return 0
    else:
        return (30-rank[0])                             ## The topn review will  get 30 as rank ( higher the better)
    
    
    
def rank_drug_train(name):
    global count 
    use_case = df.iloc[count][2]
    count = count + 1
#     print(use_case)
    a = help_dict[use_case]   ## this will create a dataframe
    rank=a.index[a['drug'] == name].tolist()
    
    if(rank[0] > 30 ):                                  ## those who has less than 30 reviews are ranked zero
        return 0
    else:
        return (30-rank[0])       
df['rank_of_drug'] = df['name_of_drug'].apply(rank_drug_train)
count =0
to_test['rank_of_drug'] = to_test['name_of_drug'].apply(rank_drug_test)

to_test.head(3)
df.head(3)

df.drop(["patient_id","name_of_drug",
        "use_case_for_drug",
        "review_by_patient","drug_approved_by_UIC","rank_of_drug"],axis=1,inplace=True)


to_test.drop(["patient_id","name_of_drug",
        "use_case_for_drug",
        "review_by_patient","drug_approved_by_UIC","rank_of_drug"],axis=1,inplace=True)
df.head(3)
to_test.head(3)
tc = df.corr() 

plt.figure(figsize=(20,16))
sns.heatmap(tc, annot = True, cmap ='plasma',  
            linecolor ='black', linewidths = 1) 

X = df.drop("base_score",axis=1)
y= df.base_score

from sklearn.preprocessing import  MinMaxScaler
sc= MinMaxScaler()
X= sc.fit_transform(X)
y= y.values.reshape(-1,1)


to_test = sc.transform(to_test)
################################

# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)

# X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)
################################

from sklearn.model_selection import train_test_split as split

X_train, X_test, y_train, y_test = split(X,y, test_size=0.01,random_state=51)#
X_train
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
# import lightgbm as lgb

from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import mean_squared_error
# XGB parameters
xgb_reg_params = {
#     'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
    'learning_rate':    hp.choice('learning_rate',    np.arange(0.15, 0.25, 0.01)),
    'n_estimators':     hp.choice('n_estimators', np.arange(1000,1800,10, dtype=int)),
    'max_depth':        hp.choice('max_depth',        np.arange(5, 15, 1, dtype=int)),
#     'gamma': hp.choice('gamma', np.arange(0, 0.4, 0.1, dtype=int)),
#     'subsample':        hp.uniform('subsample', 0.8, 1),
}


xgb_fit_params = {
    'eval_metric': 'rmse',
    'early_stopping_rounds': 10,
    'verbose': False
}


from sklearn.metrics import mean_squared_error
xgb_para = dict()
xgb_para['reg_params'] = xgb_reg_params
xgb_para['fit_params'] = xgb_fit_params
xgb_para['loss_func' ] = lambda y,pred: np.sqrt(mean_squared_error(y, pred))
# # LightGBM parameters

# # reg_params for object instantiation
# lgb_reg_params = {
#     'learning_rate':    hp.choice('learning_rate',    np.arange(0.1, 0.5, 0.05)),
#     'max_depth':        hp.choice('max_depth',        np.arange(5, 20, 1, dtype=int)),
#     'num_leaves': hp.choice('num_leaves', np.arange(16, 40, 2, dtype=int)),
# #     'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
# #     'num_leaves':        hp.choice('num_leaves', 200, 400,5, dtype=int),
#     'n_estimators':     hp.choice('n_estimators', np.arange(1500,2100,10, dtype=int)),
#     'random_state': 51,
#     'boosting_type': 'gbdt'
# }


# # fit_params for the fit() function
# lgb_fit_params = {
#     'eval_metric': 'rmse',
#     'early_stopping_rounds': 10,
#     'verbose': False
# }


# lgb_para = dict()
# lgb_para['reg_params'] = lgb_reg_params
# lgb_para['fit_params'] = lgb_fit_params
# lgb_para['loss_func' ] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
class HPOpt(object):
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test  = x_test
        self.y_train = y_train#.ravel()
        self.y_test  = y_test#.ravel()

    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials

    def xgb_reg(self, para):
        reg = xgb.XGBRegressor(**para['reg_params'])
        return self.train_reg(reg, para)
    
    def lgb_reg(self, para):
        reg = lgb.LGBMRegressor(**para['reg_params'])
        return self.train_reg(reg, para)


    def train_reg(self, reg, para):
        reg.fit(self.x_train, self.y_train,
                eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                **para['fit_params'])
        pred = reg.predict(self.x_test)
        loss = para['loss_func'](self.y_test, pred)
        return {'loss': loss, 'status': STATUS_OK}

obj = HPOpt(X_train, X_test, y_train, y_test)
xgb_opt = obj.process(fn_name='xgb_reg', space=xgb_para, trials=Trials(), algo=tpe.suggest, max_evals=100)
print (xgb_opt)
model = xgb.XGBRegressor( #colsample_bytree = 0.7,
                         learning_rate=0.17,
                         n_estimators =1400,
                         max_depth =5,
#                          min_child_weight=0,
#                          subsample=0.895795844955654,
#                          gamma=0.1,
    
                        )
model.fit(X_train, y_train)#.ravel()
y_pred = model.predict(X_test)


from sklearn import metrics

print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)   # best = 1
# lgb_opt = obj.process(fn_name='lgb_reg', space=lgb_para, trials=Trials(), algo=tpe.suggest, max_evals=100)
# print (lgb_opt)
# 0.9996468659750317
# l_obj = lgb.LGBMRegressor(boosting_type='gbdt',
#                             num_leaves=30,
#                             max_depth=13,
#                             learning_rate=0.35,
#                             n_estimators=2100,
#                             random_state=51,
#                             n_jobs=-1,
# #                             silent=-1,
#                          )

# l_obj.fit(X_train, y_train.ravel())
# y_pred = l_obj.predict(X_test)


# print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
# from sklearn.metrics import r2_score
# r2_score(y_test, y_pred)   # best = 1
# base_learners = [
#                  ('rf_1', xgb.XGBRegressor( n_estimators =1160,
#                          colsample_bytree = 0.7,
#                          learning_rate=0.2,
#                          max_depth =7,
#                          min_child_weight=0,
#                          subsample=0.8380905249423865,
# #                          gamma=0,
    
#                         )),
    
    
#                        ('bharat', lgb.LGBMRegressor(boosting_type='gbdt',
#                             num_leaves=30,
#                             max_depth=13,
#                             learning_rate=0.35,
#                             n_estimators=2100,
#                             random_state=51,
#                             n_jobs=-1,
# #                             silent=-1,
#                          ))    
#                 ]
# from sklearn.ensemble import StackingRegressor
# from sklearn.ensemble import RandomForestRegressor
# reg = StackingRegressor(estimators=base_learners, 
#                          final_estimator=RandomForestRegressor(n_estimators=10,
#                                            random_state=42)
#                         )
# reg.fit(X_train, y_train.ravel())
# y_pred = reg.predict(X_test)


# print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
# from sklearn.metrics import r2_score
# r2_score(y_test, y_pred)   # best = 1
to_test

# model = xgb.XGBRegressor( #colsample_bytree = 0.7,
#                          learning_rate=0.17,
#                          n_estimators =1400,
#                          max_depth =5,
# #                          min_child_weight=0,
# #                          subsample=0.895795844955654,
# #                          gamma=0.1,
    
#                         )
# model.fit(X_train, y_train)#.ravel()
# y_pred = model.predict(X_test)


# from sklearn import metrics

# print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
result = model.predict(to_test)
result.max()
result.min()
result.mean()
temp_df=pd.read_csv("/kaggle/input/hackerearth-effectiveness-of-std-drugs/dataset/test.csv")

result_data=temp_df['patient_id']

result_data = pd.DataFrame(result_data)

result_data['base_score']=result

result_data.to_csv('std_95.csv',index=False) 

result_data.head()
