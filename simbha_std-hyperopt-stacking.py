# Regression
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error as mse

import xgboost as xgb
import lightgbm as lgb
import catboost as cat

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import os
print(os.listdir('../input'))
train = pd.read_csv("../input/std-drug/train.csv")
test = pd.read_csv("../input/std-drug/test.csv")
min_rating = train.effectiveness_rating.min()
max_rating = train.effectiveness_rating.max()

def scale_rating(rating):
    # Sacling from (1,10) to (0,5) and then replacing 0,1,2 in ratings with 0 (poor) and 3,4,5 with 1 (good).
    rating -= min_rating
    rating = rating/(max_rating - 1)
    rating *= 3
    rating = int(round(rating,0))
    return rating

train['new_effect_score'] = train.effectiveness_rating.apply(scale_rating)
test['new_effect_score'] = test.effectiveness_rating.apply(scale_rating)
from nltk.corpus import wordnet

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

# clean text data
train["review_by_patient"] = train["review_by_patient"].apply(lambda x: clean_text(x))

test["review_by_patient"] = test["review_by_patient"].apply(lambda x: clean_text(x))
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
train["sentiments"] = train["review_by_patient"].apply(lambda x: sid.polarity_scores(x))
test["sentiments"] = test["review_by_patient"].apply(lambda x: sid.polarity_scores(x))

# reviews_df = pd.concat([reviews_df.drop(['sentiments'], axis=1), reviews_df['sentiments'].apply(pd.Series)], axis=1)

train.sentiments[0]
# train['neg_sentiment'] = train.sentiments.apply(lambda x:round(x['neg']))
train['pos_sentiment'] = train.sentiments.apply(lambda x:round(x['pos']))
# train['neu_sentimant'] = train.sentiments.apply(lambda x:round(x['neu']))

test['neg_sentiment'] = test.sentiments.apply(lambda x:round(x['neg']))
test['pos_sentiment'] = test.sentiments.apply(lambda x:round(x['pos']))
test['neu_sentimant'] = test.sentiments.apply(lambda x:round(x['neu']))
train.head()
feat = train.columns.to_list()
target='base_score'
delete = ['patient_id','name_of_drug','use_case_for_drug','review_by_patient','drug_approved_by_UIC',target,'sentiments','neg_sentiment','neu_sentimant']
for i in delete:
    feat.remove(i)
train.corr()
# from sklearn.preprocessing import MinMaxScaler
# scalar = MinMaxScaler()
# X=scalar.fit_transform(train[feat])
# Y = train[target].values.reshape(-1,1)

def baseliner(X,Y):
    print("Baseliner Models(All)")
    eval_dict = {}
    models = [
#         KNeighborsRegressor(), GaussianNB(), 
        lgb.LGBMRegressor(), ExtraTreesRegressor(), xgb.XGBRegressor(objective='reg:squarederror'), 
        cat.CatBoostRegressor(verbose=0), GradientBoostingRegressor(), RandomForestRegressor(), 
        LinearRegression(), DecisionTreeRegressor(), ExtraTreeRegressor(), AdaBoostRegressor(), 
#         BaggingRegressor(), ElasticNet(), Lasso(), Ridge(),SVR(),
        ]  

    print("sklearn Model Name  \t  rmse")
    print("--" * 50)
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=42)
    for i in models:
        model = i
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        model_name =str(i).split("(")[0]
        result=np.sqrt(mse(y_test,y_pred))
        print(f"{model_name} \t {result}")
baseliner(train[feat],train[target])
from hyperopt import hp
import numpy as np
from sklearn.metrics import mean_squared_error


# XGB parameters
xgb_reg_params = {
    'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 0.31, 0.05)),
    'max_depth':        hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
    'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
    'subsample':        hp.uniform('subsample', 0.8, 1),
    'n_estimators':     500,
}
xgb_fit_params = {
    'eval_metric': 'rmse',
    'early_stopping_rounds': 10,
    'verbose': False
}
xgb_para = dict()
xgb_para['reg_params'] = xgb_reg_params
xgb_para['fit_params'] = xgb_fit_params
xgb_para['loss_func' ] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))

# LightGBM parameters
lgb_reg_params = {
    'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 0.31, 0.05)),
    'max_depth':        hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
    'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
    'subsample':        hp.uniform('subsample', 0.8, 1),
    'n_estimators':     100,
}
lgb_fit_params = {
    'eval_metric': 'l2',
    'early_stopping_rounds': 10,
    'verbose': False
}
lgb_para = dict()
lgb_para['reg_params'] = lgb_reg_params
lgb_para['fit_params'] = lgb_fit_params
lgb_para['loss_func' ] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))


# CatBoost parameters
ctb_reg_params = {
    'learning_rate':     hp.choice('learning_rate',     np.arange(0.05, 0.31, 0.05)),
    'max_depth':         hp.choice('max_depth',         np.arange(5, 16, 1, dtype=int)),
    'colsample_bylevel': hp.choice('colsample_bylevel', np.arange(0.3, 0.8, 0.1)),
    'n_estimators':      100,
    'eval_metric':       'RMSE',
}
ctb_fit_params = {
    'early_stopping_rounds': 10,
    'verbose': False
}
ctb_para = dict()
ctb_para['reg_params'] = ctb_reg_params
ctb_para['fit_params'] = ctb_fit_params
ctb_para['loss_func' ] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))

import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials


class HPOpt(object):

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test  = x_test
        self.y_train = y_train
        self.y_test  = y_test

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

    def ctb_reg(self, para):
        reg = ctb.CatBoostRegressor(**para['reg_params'])
        return self.train_reg(reg, para)

    def train_reg(self, reg, para):
        reg.fit(self.x_train, self.y_train,
                eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                **para['fit_params'])
        pred = reg.predict(self.x_test)
        loss = para['loss_func'](self.y_test, pred)
        return {'loss': loss, 'status': STATUS_OK}

x_train, x_test, y_train, y_test = train_test_split(train[feat],train[target], test_size=0.3, random_state=42)
obj = HPOpt(x_train, x_test, y_train, y_test)
xgb_opt = obj.process(fn_name='xgb_reg', space=xgb_para, trials=Trials(), algo=tpe.suggest, max_evals=200)
lgb_opt = obj.process(fn_name='lgb_reg', space=lgb_para, trials=Trials(), algo=tpe.suggest, max_evals=200)
ctb_opt = obj.process(fn_name='ctb_reg', space=ctb_para, trials=Trials(), algo=tpe.suggest, max_evals=200)
from sklearn.ensemble import StackingRegressor
# Creating First Layer
base_learners = [
    ('lgbm',lgb.LGBMRegressor(boosting_type='gbdt',
                        num_leaves=300,
                        max_depth=16,
                        learning_rate = 0.5,
                        n_estimators=400,
                         subsample_for_bin=200000,
#                          class_weight=,
#                          min_split_gain=0.01,
#                          min_child_weight=0.1,
                         objective=None,
                         min_child_samples=20,
                         subsample=1.0,
                         subsample_freq=0,
                         colsample_bytree=1.0,
                         reg_alpha=0.0,
                         reg_lambda=0.0,
                         random_state=None,
                         n_jobs=-1,
                         silent=True
                        )),
]
# Initializating Stacking Regressor with the meta lerner
reg = StackingRegressor(estimators=base_learners,final_estimator=xgb.XGBRegressor(objective='reg:squarederror',n_estimators=500,max_depth=8,learning_rate=0.25,colsample_bytree=0.7,min_child_weight=1,subsample=0.8862306230110487))
train_X, test_X, train_y, test_y = train_test_split(train[feat],train[target], test_size=0.3, random_state=42)
reg.fit(train_X,train_y)
y_pred = reg.predict(test_X)
print(np.sqrt(mse(test_y,y_pred)))


lgbm = lgb.LGBMRegressor(boosting_type='gbdt',
                        num_leaves=300,
                        max_depth=16,
                        learning_rate = 0.5,
                        n_estimators=400,
                         subsample_for_bin=200000,
#                          class_weight=,
#                          min_split_gain=0.01,
#                          min_child_weight=0.1,
                         objective=None,
                         min_child_samples=20,
                         subsample=1.0,
                         subsample_freq=0,
                         colsample_bytree=1.0,
                         reg_alpha=0.0,
                         reg_lambda=0.0,
                         random_state=None,
                         n_jobs=-1,
                         silent=True
                        )
lgbm.fit(train[feat],train[target])
train_X, test_X, train_y, test_y = train_test_split(train[feat],train[target], test_size=0.3, random_state=42)
y_pred = lgbm.predict(test_X)
print(np.sqrt(mse(test_y,y_pred)))
train_X, test_X, train_y, test_y = train_test_split(train[feat],train[target], test_size=0.3, random_state=42)

my_model = xgb.XGBRegressor(objective='reg:squarederror',n_estimators=500,max_depth=8,learning_rate=0.25,colsample_bytree=0.7,min_child_weight=1,subsample=0.8862306230110487)
my_model.fit(train[feat],train[target],early_stopping_rounds=10, eval_set=[(test_X, test_y)], verbose=1)
y_pred = my_model.predict(test_X)
print(np.sqrt(mse(test_y,y_pred)))
y_pred = my_model.predict(test[feat])
ans=[]
for i in range(len(y_pred)):
    ans.append(round(y_pred[i],2))
sub = test['patient_id']
sub = pd.DataFrame(sub)
sub['base_score']=ans
#Creating Link in Kaggle
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

def create_download_link(df, title = "Download CSV file", filename = "std_20.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(sub)

