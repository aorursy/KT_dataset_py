%load_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.tabular import *



from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold

import time



import lightgbm as lgb

import xgboost as xgb



import ast



from sklearn.metrics import mean_squared_error



import seaborn as sns



from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.decomposition import TruncatedSVD
PATH = "../input/tmdb-box-office-prediction/"
train = pd.read_csv(f'{PATH}train.csv', parse_dates=['release_date'])

test = pd.read_csv(f'{PATH}test.csv', parse_dates=['release_date'])
train_votes = pd.read_csv('../input/tmdb-prediction-votes/trainRatingTotalVotes.csv')

test_votes = pd.read_csv('../input/tmdb-prediction-votes/testRatingTotalVotes.csv')

train = pd.merge(train, train_votes, how='left', on=['imdb_id'])

test = pd.merge(test, test_votes, how='left', on=['imdb_id'])
dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast']



def text_to_dict(df):

    for column in dict_columns:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df

        

train = text_to_dict(train)

test = text_to_dict(test)
def build_category_list(x, field, feature):

    regex = re.compile('[^0-9a-zA-Z_]')

    category_list = ""

    

    

    for d in x:

        new_category = regex.sub('', d[field].lower().replace(" ","_"))

        

        # Exception for cast: keep only 0 and 1 to limit nb of values

        #        if feature == 'cast' and d['order'] > 1:

        #            pass

        #        else:

        category_list += new_category + ","

    return category_list.strip().strip(",").split(",")





target_fields = {'belongs_to_collection': 'name', 'genres': 'name',

                 'production_countries': 'iso_3166_1', 'production_companies': 'name',

                 'spoken_languages': 'iso_639_1', 'Keywords': 'name', 'cast': 'name'

                }



for k,v in target_fields.items():

    train[k] = train[k].apply(lambda x: build_category_list(x, v, k))

    test[k] = test[k].apply(lambda x: build_category_list(x, v, k))

    

    

target_fields = {'cast':{'field':'name', 'role_field':'order', 'role_values':[0,1,2]}}
train.loc[train['id'] == 16,'revenue'] = 192864          # Skinning

train.loc[train['id'] == 90,'budget'] = 30000000         # Sommersby          

train.loc[train['id'] == 118,'budget'] = 60000000        # Wild Hogs

train.loc[train['id'] == 149,'budget'] = 18000000        # Beethoven

train.loc[train['id'] == 313,'revenue'] = 12000000       # The Cookout 

train.loc[train['id'] == 451,'revenue'] = 12000000       # Chasing Liberty

train.loc[train['id'] == 464,'budget'] = 20000000        # Parenthood

train.loc[train['id'] == 470,'budget'] = 13000000        # The Karate Kid, Part II

train.loc[train['id'] == 513,'budget'] = 930000          # From Prada to Nada

train.loc[train['id'] == 797,'budget'] = 8000000         # Welcome to Dongmakgol

train.loc[train['id'] == 819,'budget'] = 90000000        # Alvin and the Chipmunks: The Road Chip

train.loc[train['id'] == 850,'budget'] = 90000000        # Modern Times

train.loc[train['id'] == 1112,'budget'] = 7500000        # An Officer and a Gentleman

train.loc[train['id'] == 1131,'budget'] = 4300000        # Smokey and the Bandit   

train.loc[train['id'] == 1359,'budget'] = 10000000       # Stir Crazy 

train.loc[train['id'] == 1542,'budget'] = 1              # All at Once

train.loc[train['id'] == 1542,'budget'] = 15800000       # Crocodile Dundee II

train.loc[train['id'] == 1571,'budget'] = 4000000        # Lady and the Tramp

train.loc[train['id'] == 1714,'budget'] = 46000000       # The Recruit

train.loc[train['id'] == 1721,'budget'] = 17500000       # Cocoon

train.loc[train['id'] == 1865,'revenue'] = 25000000      # Scooby-Doo 2: Monsters Unleashed

train.loc[train['id'] == 2268,'budget'] = 17500000       # Madea Goes to Jail budget

train.loc[train['id'] == 2491,'revenue'] = 6800000       # Never Talk to Strangers

train.loc[train['id'] == 2602,'budget'] = 31000000       # Mr. Holland's Opus

train.loc[train['id'] == 2612,'budget'] = 15000000       # Field of Dreams

train.loc[train['id'] == 2696,'budget'] = 10000000       # Nurse 3-D

train.loc[train['id'] == 2801,'budget'] = 10000000       # Fracture
test.loc[test['id'] == 3889,'budget'] = 15000000       # Colossal

test.loc[test['id'] == 6733,'budget'] = 5000000        # The Big Sick

test.loc[test['id'] == 3197,'budget'] = 8000000        # High-Rise

test.loc[test['id'] == 6683,'budget'] = 50000000       # The Pink Panther 2

test.loc[test['id'] == 5704,'budget'] = 4300000        # French Connection II

test.loc[test['id'] == 6109,'budget'] = 281756         # Dogtooth

test.loc[test['id'] == 7242,'budget'] = 10000000       # Addams Family Values

test.loc[test['id'] == 7021,'budget'] = 17540562       #  Two Is a Family

test.loc[test['id'] == 5591,'budget'] = 4000000        # The Orphanage

test.loc[test['id'] == 4282,'budget'] = 20000000       # Big Top Pee-wee
power_six = train.id[train.budget > 1000][train.revenue < 100]



for k in power_six :

    train.loc[train['id'] == k,'revenue'] =  train.loc[train['id'] == k,'revenue'] * 1000000
def multi_hot_encode(df, column_name, mlb=MultiLabelBinarizer()):

    encoded = pd.DataFrame(mlb.fit_transform(df[column_name]))

    encoded.columns = [f'{column_name}_{i}'.format(i) for i in mlb.classes_]

    return mlb, encoded
def conv_column_to_SVD(df, column_name, size=10, sparse_mlb=MultiLabelBinarizer(sparse_output=True)):

    #sparse_mlb = MultiLabelBinarizer(sparse_output=True)

    sparse_matrix = sparse_mlb.fit_transform(df[column_name])

    sparse_SVD = TruncatedSVD(size)

    sparse_TSVD = pd.DataFrame(sparse_SVD.fit_transform(sparse_matrix))

    sparse_TSVD.columns = [f'{column_name}_{i}' for i in range(size)]

    return sparse_mlb, sparse_TSVD
genre_mlb, genre_encoded = multi_hot_encode(train,'genres')

cast_sparse_mlb, cast_TSVD = conv_column_to_SVD(train, 'cast', 10)

keywords_sparse_mlb, keywords_TSVD = conv_column_to_SVD(train, 'Keywords', 10)

languages_mlb, languages_TSVD = conv_column_to_SVD(train, 'spoken_languages', 5)

prodcomp_mlb, prodcomp_TSVD = conv_column_to_SVD(train, 'production_companies', 5)

prodcountries_mlb, prodcountries_TSVD = conv_column_to_SVD(train, 'production_countries', 3)
_,test_genre= multi_hot_encode(test,'genres',genre_mlb)

_,test_cast = conv_column_to_SVD(test, 'cast', 10)

_,test_keywords = conv_column_to_SVD(test, 'Keywords', 10)

_,test_languages = conv_column_to_SVD(test, 'spoken_languages', 5)

_,test_prodcomp = conv_column_to_SVD(test, 'production_companies', 5)

_,test_prodcountries = conv_column_to_SVD(test, 'production_countries', 3)
train = train.join(genre_encoded)

train = train.join(cast_TSVD)

train = train.join(keywords_TSVD)

train = train.join(languages_TSVD)

train = train.join(prodcomp_TSVD)

train = train.join(prodcountries_TSVD)
test = test.join(test_genre)

test = test.join(test_cast)

test = test.join(test_keywords)

test = test.join(test_languages)

test = test.join(test_prodcomp)

test = test.join(test_prodcountries)
add_datepart(train, 'release_date')

add_datepart(test, 'release_date')
cats_to_drop=['id','title','genres', 'cast', 'crew','Keywords','spoken_languages','production_companies','production_countries','homepage', 'belongs_to_collection','poster_path','imdb_id','original_language', 'original_title', 'overview','tagline','status']

train = train.drop(cats_to_drop, axis=1)

train = train.drop('genres_tv_movie', axis=1)

test = test.drop(cats_to_drop,axis=1)
print(train.shape)

print(test.shape)
train["revenue"]=np.log(train["revenue"]).astype('float')

train["budget"]=np.log(train["budget"]+0.1).astype('float')

test["budget"]=np.log(test["budget"]+0.1).astype('float')
train.loc[train["runtime"].isnull(),"runtime"]=train["runtime"].mode()[0]

train.loc[train["rating"].isnull(),"rating"]=train["rating"].mode()[0]

train.loc[train["totalVotes"].isnull(),"totalVotes"]=train["totalVotes"].mode()[0]



test.loc[test["runtime"].isnull(),"runtime"]=train["runtime"].mode()[0]

test.loc[test["rating"].isnull(),"rating"]=train["runtime"].mode()[0]

test.loc[test["totalVotes"].isnull(),"totalVotes"]=train["totalVotes"].mode()[0]



test.loc[test["release_Year"].isnull(),"release_Year"]=train["release_Year"].mode()[0]

test.loc[test["release_Month"].isnull(),"release_Month"]=train["release_Month"].mode()[0]

test.loc[test["release_Week"].isnull(),"release_Week"]=train["release_Week"].mode()[0]

test.loc[test["release_Day"].isnull(),"release_Day"]=train["release_Day"].mode()[0]

test.loc[test["release_Dayofweek"].isnull(),"release_Dayofweek"]=train["release_Dayofweek"].mode()[0]

test.loc[test["release_Dayofyear"].isnull(),"release_Dayofyear"]=train["release_Dayofyear"].mode()[0]
X = train.drop(['revenue'], axis=1)

y = train['revenue']

X_test = test

n_fold = 5

folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

repeated_folds = RepeatedStratifiedKFold(n_splits=10, n_repeats=20, random_state=42)



# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)

# X_test = scaler.transform(X_test)
def train_model(X, X_test, y, params, folds, model_type='lgb', plot_feature_importance=False, averaging='usual', model=None):

    oof = np.zeros(len(X))

    prediction = np.zeros(len(X_test))

    scores = []

    feature_importance = pd.DataFrame()

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):

        print('Fold', fold_n, 'started at', time.ctime())

        X_train, X_valid = X.loc[train_index], X.loc[valid_index]

        y_train, y_valid = y[train_index], y[valid_index]

        

        if model_type == 'lgb':

            train_data = lgb.Dataset(X_train, label=y_train)

            valid_data = lgb.Dataset(X_valid, label=y_valid)

            

            model = lgb.train(params,

                    train_data,

                    num_boost_round=20000,

                    valid_sets = [train_data, valid_data],

                    verbose_eval=1000,

                    early_stopping_rounds = 200)

            

            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test, num_iteration=model.best_iteration)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X_train.columns)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X_train.columns)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X_train.columns), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_train.columns), ntree_limit=model.best_ntree_limit)

        

        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            y_pred_valid = model.predict_proba(X_valid).reshape(-1,)

            score = mean_squared_error(y_valid, y_pred_valid)

            # print(f'Fold {fold_n}. AUC: {score:.4f}.')

            # print('')

            

            y_pred = model.predict_proba(X_test)[:, 1]

            

        if model_type == 'glm':

            model = sm.GLM(y_train, X_train, family=sm.families.Binomial())

            model_results = model.fit()

            model_results.predict(X_test)

            y_pred_valid = model_results.predict(X_valid).reshape(-1,)

            score = mean_squared_error(y_valid, y_pred_valid)

            

            y_pred = model_results.predict(X_test)

            

        if model_type == 'cat':

            model = CatBoostClassifier(iterations=20000, learning_rate=0.1, loss_function='Logloss',  eval_metric='AUC', **params)

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict_proba(X_valid)

            y_pred = model.predict_proba(X_test)

            

        oof[valid_index] = y_pred_valid.reshape(-1,)

        scores.append(mean_squared_error(y_valid, y_pred_valid))



        if averaging == 'usual':

            prediction += y_pred

        elif averaging == 'rank':

            prediction += pd.Series(y_pred).rank().values  

        

        if model_type == 'lgb':

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = X.columns

            fold_importance["importance"] = model.feature_importance()

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= n_fold

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    if model_type == 'lgb':

        feature_importance["importance"] /= n_fold

        if plot_feature_importance:

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

        

            return oof, prediction, feature_importance

        return oof, prediction, scores

    

    else:

        return oof, prediction, scores
params = {'num_leaves': 16,

         'min_data_in_leaf': 2,

         'objective': 'regression',

         'max_depth': 20,

         'learning_rate': 0.008,

         'boosting': 'gbdt',

         'bagging_freq': 5,

         'feature_fraction': 0.82,

         'bagging_seed': 11,

         'reg_alpha': 1.7,

         'reg_lambda': 6,

         'random_state': 42,

         'metric': 'mse',

         'verbosity': -1,

         'subsample': 0.81,

         'min_gain_to_split': 0.01,

         'min_child_weight': 10,

         'num_threads': 8}

oof_lgb, prediction_lgb, scores = train_model(X, X_test, y, params=params, folds=folds, model_type='lgb', plot_feature_importance=True)
xgb_params = {'eta': 0.1, 'max_depth': 3, 'subsample': 0.9, 'colsample_bytree': 0.9, 

          'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 8}

oof_xgb, prediction_xgb, scores = train_model(X, X_test, y, params=xgb_params, folds=folds, model_type='xgb')
sns.distplot(prediction_lgb, hist=False) # blue

sns.distplot(prediction_xgb, hist=False) # orange

sns.distplot(train['revenue'], hist=False) #green
train['revenue'].mean()/((prediction_lgb+prediction_xgb)/2).mean()
submission = pd.read_csv(f'{PATH}sample_submission.csv')

submission['revenue'] = np.exp(0.50*(prediction_lgb+prediction_xgb))

submission.to_csv(f'submission.csv', index=False)