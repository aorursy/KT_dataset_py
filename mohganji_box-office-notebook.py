### READING DATA FROM INPUT



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# remove warning!!

pd.options.mode.chained_assignment = None  # default='warn'





import os

# print(os.listdir("../input"))

df = pd.read_csv("../input/train.csv", index_col=[0])

testData = pd.read_csv("../input/test.csv")

### DESCRIPTION OF DATA

df.describe(include="all")
### DATA PREPARATION ###

def prepare(df):

    global json_cols

    global train_dict



    ### separate release date into year, month, day

    df[['release_month','release_day','release_year']]=df['release_date'].str.split('/',expand=True).replace(np.nan, 0).astype(int)

    df['release_year'] = df['release_year']

    df.loc[ (df['release_year'] <= 18) & (df['release_year'] < 100), "release_year"] += 2000

    df.loc[ (df['release_year'] > 18)  & (df['release_year'] < 100), "release_year"] += 1900

    ###



    ### add ratio datas for correlations

    meanRuntime = df['runtime'].mean()

    df.loc[ df['runtime'] == 0 ,'runtime'] = meanRuntime

    df['_budget_runtime_ratio'] =  df['budget']/df['runtime']

    df['_budget_popularity_ratio'] = df['budget']/df['popularity']

    df['_budget_year_ratio'] = df['budget']/(df['release_year'])

    df['_releaseYear_popularity_ratio'] = df['release_year']/df['popularity']

    df['_popularity_releaseYear_ratio'] = df['popularity']/df['release_year']

    ###



    ### add binary fields for descriptive data

    df['isMovieReleased'] = 1

    df.loc[ df['status'] != "Released" ,"isMovieReleased"] = 0 



    df['isOriginalLanguageEng'] = 0 

    df.loc[ df['original_language'] == "en" ,"isOriginalLanguageEng"] = 1

    ###



    ### add mean data

    df['meanRuntimeByYear'] = df.groupby("release_year")["runtime"].aggregate('mean')

    ###

    

    df = df.drop(['belongs_to_collection','genres','homepage','overview', 'imdb_id'

    ,'poster_path','production_companies','production_countries','release_date','spoken_languages'

    ,'status','title','Keywords','cast','crew','original_language','original_title','tagline'

    ],axis=1)

    

    df.fillna(value=0.0, inplace = True) 

    df.fillna(value=0.0, inplace = True) 

    return df



# all_data = prepare(pd.concat([df, testData], sort=True).reset_index(drop = True))

train = prepare(df)

train['id'] = range(1, len(train) + 1)

train.index = train['id']



test = prepare(testData)

test['id'] = range(1, len(test) + 1)

test.index = test['id']





### END OF DATA PREPARATION ###

# data after preparation:

features = list(train.columns)

features =  [i for i in features if i != 'id' and i != 'revenue']

print("*** Features are:")

print(train.dtypes)

train.describe(include="all")
### K-FOLD Class Decleration

from sklearn.model_selection import GroupKFold

from sklearn.metrics import mean_squared_error



def score(data, y):

    validation_res = pd.DataFrame(

    {"id": data["id"].values,

     "transactionrevenue": data["revenue"].values,

     "predictedrevenue": np.expm1(y)})



    validation_res = validation_res.groupby("id")["transactionrevenue", "predictedrevenue"].sum().reset_index()

    return np.sqrt(mean_squared_error(np.log1p(validation_res["transactionrevenue"].values), 

                                     np.log1p(validation_res["predictedrevenue"].values)))



class KFoldValidation():

    def __init__(self, data, n_splits=5):

        unique_vis = np.array(sorted(data['id'].astype(str).unique()))

        folds = GroupKFold(n_splits)

        ids = np.arange(data.shape[0])

        

        self.fold_ids = []

        for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):

            self.fold_ids.append([

                    ids[data['id'].astype(str).isin(unique_vis[trn_vis])],

                    ids[data['id'].astype(str).isin(unique_vis[val_vis])]

                ])

            

    def validate(self, train, test, features, model, name="", prepare_stacking=False, 

                 fit_params={"early_stopping_rounds": 500, "verbose": 0, "eval_metric": "rmse"}):

        model.FI = pd.DataFrame(index=features)

        full_score = 0

        

        if prepare_stacking:

            test[name] = 0

            train[name] = np.NaN

        

        for fold_id, (trn, val) in enumerate(self.fold_ids):

            devel = train[features].iloc[trn]

            y_devel = np.log1p(train["revenue"].iloc[trn])

            valid = train[features].iloc[val]

            y_valid = np.log1p(train["revenue"].iloc[val])

                       

            #print("Fold ", fold_id, ":")

            model.fit(devel, y_devel, eval_set=[(valid, y_valid)], **fit_params)

            

            if len(model.feature_importances_) == len(features):  # some bugs in catboost?

                model.FI['fold' + str(fold_id)] = model.feature_importances_ / model.feature_importances_.sum()



            predictions = model.predict(valid)

            predictions[predictions < 0] = 0

            #print("Fold ", fold_id, " error: ", mean_squared_error(y_valid, predictions)**0.5)

            

            fold_score = score(train.iloc[val], predictions)

            full_score += fold_score / len(self.fold_ids)

            #print("Fold ", fold_id, " score: ", fold_score)

            

            if prepare_stacking:

                train[name].iloc[val] = predictions

                

                test_predictions = model.predict(test[features])

                test_predictions[test_predictions < 0] = 0

                test[name] += test_predictions / len(self.fold_ids)

                

        print("Final score: ", full_score)

        return full_score



KfoldResult = KFoldValidation(train)
### K-Fold Validation with lightgbm

import lightgbm as lgb



lgbmodel = lgb.LGBMRegressor(n_estimators=10000, 

                             objective="regression", 

                             metric="rmse", 

                             num_leaves=20, 

                             min_child_samples=100,

                             learning_rate=0.01, 

                             bagging_fraction=0.8, 

                             feature_fraction=0.8, 

                             bagging_frequency=1, 

                             bagging_seed=2019, 

                             subsample=.9, 

                             colsample_bytree=.9,

                             use_best_model=True)



KfoldResult.validate(train, test, features , lgbmodel, name="lgbfeatures", prepare_stacking=True) 

### plot features importance

sortedFeatureValues = lgbmodel.FI.mean(axis=1).sort_values()

sortedFeatureValues.plot(kind="barh",title = "Features Importance", figsize = (10,10));

### Correlation matrix

from pandas.plotting import scatter_matrix



def plot_corr(df,size=10):

    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.



    Input:

        df: pandas DataFrame

        size: vertical and horizontal size of the plot'''



    corr = df.corr()

    fig = plt.figure(figsize=(size, size))

    ax = fig.add_subplot(111)

    cmap = plt.cm.get_cmap('RdYlBu')

    cax = ax.matshow(corr, vmin=-1, vmax=1, cmap=cmap)

    fig.colorbar(cax)

    ax.matshow(corr, cmap=cmap)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical');

    plt.yticks(range(len(corr.columns)), corr.columns);

    plt.show()



### WAY 1 ###

plot_corr(train, 10)



### WAY 2 ###

#corr = train.corr()

#corr.style.background_gradient(cmap='coolwarm').set_precision(2)



### WAY 3 ###

# scatter_matrix(train)
def trainWithModel(X, Y, testX, features, target, model, modelName):    

    predictY = model.predict(testX)



    for feature in features:

        testResults = pd.DataFrame(data=predictY, columns=[target])

        testResults[feature] = test[feature]

        testResults = testResults.sort_values(feature)

        plt.xlabel(feature)

        plt.ylabel(target)

        plt.scatter(train[feature], Y, color='black', s=3)    

        plt.plot(testResults[feature], testResults[target], color='blue', linewidth=1)

        plt.show()



    predictY = pd.DataFrame(data=predictY,    # values

                  columns=[target])  # 1st row as the column names

    predictY['id'] = testData['id']

    predictY.index = predictY['id']



    predictY.to_csv(modelName+".csv", index=False)
### LINEAR REGRESSION TRAINING

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression



# filterItems = ['release_year', 'runtime']#, 'popularity', '_budget_runtime_ratio', '_budget_popularity_ratio', '_releaseYear_popularity_ratio', 'budget']

testItem = 'revenue'



## features based on correlation matrix

filterItems = ['budget', '_budget_year_ratio', '_budget_runtime_ratio', 'runtime', 'release_year']



testCols = test[filterItems]

testX = testCols.values.reshape(-1, len(filterItems))



trainCols = train[filterItems]

X = trainCols.values.reshape(-1, len(filterItems))

Y = train[testItem].values



linearModel = LinearRegression()

linearModel.fit(X, Y)



trainWithModel(X, Y, testX, filterItems, testItem, linearModel, 'linearRegression')

### Ridge regression

from sklearn.linear_model import RidgeCV



# filterItems = ['release_year', 'runtime', 'popularity', '_budget_runtime_ratio', '_budget_popularity_ratio', '_releaseYear_popularity_ratio', 'budget']

filterItems = [

'budget'

,'popularity'

,'runtime'

,'release_month'

,'release_day'

,'release_year'

,'_budget_runtime_ratio'

,'_budget_popularity_ratio'

,'_budget_year_ratio'

,'_releaseYear_popularity_ratio'

# ,'_popularity_releaseYear_ratio'

# ,'isMovieReleased'

# ,'isOriginalLanguageEng'

# ,'meanRuntimeByYear'

### Last Four Items contain ill-conditioned data

]





target = 'revenue'





testCols = test[filterItems]

testX = testCols.values.reshape(-1, len(filterItems))



trainCols = train[filterItems]

X = trainCols.values.reshape(-1, len(filterItems))

Y = train[testItem].values



ridgeModel = RidgeCV(cv=5)

ridgeModel.fit(X, Y)



trainWithModel(X, Y, testX, filterItems, target, ridgeModel, 'ridgeRegression')
### Ridge regression and Lasso regression

from sklearn.linear_model import LassoCV



# filterItems = ['release_year', 'runtime', 'popularity', '_budget_runtime_ratio', '_budget_popularity_ratio', '_releaseYear_popularity_ratio', 'budget']

filterItems = [

'budget'

,'popularity'

,'runtime'

,'release_month'

,'release_day'

,'release_year'

,'_budget_runtime_ratio'

,'_budget_popularity_ratio'

,'_budget_year_ratio'

,'_releaseYear_popularity_ratio'

# ,'_popularity_releaseYear_ratio'

# ,'isMovieReleased'

# ,'isOriginalLanguageEng'

# ,'meanRuntimeByYear'

### Last Four Items contain ill-conditioned data

]



target = 'revenue'





testCols = test[filterItems]

testX = testCols.values.reshape(-1, len(filterItems))



trainCols = train[filterItems]

X = trainCols.values.reshape(-1, len(filterItems))

Y = train[testItem].values



lassoModel = LassoCV(cv=5)

lassoModel.fit(X, Y)



trainWithModel(X, Y, testX, filterItems, target, lassoModel, 'lassoRegression')