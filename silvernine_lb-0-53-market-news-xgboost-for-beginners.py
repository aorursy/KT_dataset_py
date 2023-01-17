# Import some libraries
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # graphing
import os
from datetime import datetime, timedelta # Used to subtract days from a date

print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

# Import environment
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
# Import training dataset
(market_train_df, news_train_df) = env.get_training_data()
# Process news data function
def process_news_date(news_train_df):
    # Define which columns I don't want - I just used my intuition to select these columns
    news_columns_to_drop = ['firstCreated','sourceId','headline','takeSequence','provider','subjects','audiences','bodySize','companyCount','headlineTag','sentenceCount','assetCodes','firstMentionSentence','noveltyCount12H','noveltyCount24H','noveltyCount3D','noveltyCount5D','noveltyCount7D','volumeCounts12H','volumeCounts24H','volumeCounts3D','volumeCounts5D','volumeCounts7D']
    # Drop the columns chosen from above
    news_train_df.drop(columns=news_columns_to_drop,inplace=True)
    # Create sentiment word ratio from sentimentWordCount and wordCount <- i think this feature is helpful.
    news_train_df['sentimentWordRatio'] = news_train_df['sentimentWordCount']/news_train_df['wordCount']
    # Drop sentimentWordCount and wordCount since they are incorporated into the new column sentimentWordRatio now
    news_columns_to_drop = ['wordCount','sentimentWordCount']
    news_train_df=news_train_df.drop(columns=news_columns_to_drop)
    #return the news dataframe
    return news_train_df
# Separate 'date' into year,month, and day. Then, add year,month, and day to the 'assetName'.
# Performing this will allow me to merge news & market data with this new 'combined_index' column
def combined_index(df):
    df['combined_index'] = (df['time'].dt.year).astype(str)+(df['time'].dt.month).astype(str)+(df['time'].dt.day).astype(str)+(df['assetName']).astype(str)
    return df

# mergy market & news data by 'combined_index'
def merge_market_news(market_df,news_df):
    # By having .mean(), it will take average of numeric values if there are duplicate news for the same 'combined_index'
    news_df = news_df.groupby('combined_index').mean()
    # merge news data to market data using the 'combined_index' we created
    market_df=market_df.merge(news_df,how='left',on='combined_index')
    # since there are more items in market data, ther are lots of rows with NaNs, and we fill them with 0 for training purposes.
    fill_na_columns = ['urgency','marketCommentary','relevance','sentimentClass','sentimentNegative','sentimentNeutral','sentimentPositive','sentimentWordRatio']
    market_df[fill_na_columns]=market_df[fill_na_columns].fillna(0)
    return market_df
# Process news data
news_train_df=process_news_date(news_train_df)
# Create 'combined_index' for news dataframe
news_train_df=combined_index(news_train_df).copy()
# Create 'combined_index' for market dataframe
market_train_df=combined_index(market_train_df).copy()
# Merge market & news data
market_train_df=merge_market_news(market_train_df,news_train_df)
# Pre-processes market data for training
def pre_process_market_data(market_train_df):
    # Let's remove outliers based on our EDA. Remove anything outside [-1,1] for 'returnsOpenNextMktres10'
    market_train_df = (market_train_df[(market_train_df['returnsOpenNextMktres10']<1) & (market_train_df['returnsOpenNextMktres10']>-1)]).copy()
    # Let's choose our features
    features = ['time','universe','volume','returnsClosePrevRaw1','returnsOpenPrevRaw1','returnsClosePrevRaw10','returnsOpenPrevRaw10','urgency','marketCommentary','relevance','sentimentClass','sentimentNegative','sentimentNeutral','sentimentPositive','sentimentWordRatio']
    x = market_train_df[features].copy()
    y = market_train_df[['returnsOpenNextMktres10','universe','time']].copy()
    return x,y

# Pre-processes market data for prediction for actual scoring. We are not provided with 'returnsOpenNextMktres10', hence no outlier removal is needed and we don't need to output target data.
def pre_process_market_data_actual_competition(market_train_df):
    # Let's choose our features
    features = ['volume','returnsClosePrevRaw1','returnsOpenPrevRaw1','returnsClosePrevRaw10','returnsOpenPrevRaw10','urgency','marketCommentary','relevance','sentimentClass','sentimentNegative','sentimentNeutral','sentimentPositive','sentimentWordRatio']
    x = market_train_df[features]
    return x
x,y=pre_process_market_data(market_train_df)
# Splits data for training. Takes out 30 days worth of data between training and validation set to prevent data leakage
def split_train_test_and_time(x,y,test_size):    
    # Splits data as specified test_size and creates a gap of 30 days between train and test. This helps data leakage so that the model doesn't know the future when training
    X_train = x[x['time']<(x['time'][int(len(x)*(1-test_size))]-timedelta(days=30))]
    y_train = y[y['time']<(y['time'][int(len(x)*(1-test_size))]-timedelta(days=30))]
    X_test = x[x['time']>x['time'][int(len(x)*(1-test_size))]]
    y_test = y[y['time']>y['time'][int(len(y)*(1-test_size))]]   
    # Features to be used
    features_no_universe = ['volume','returnsClosePrevRaw1','returnsOpenPrevRaw1','returnsClosePrevRaw10','returnsOpenPrevRaw10','urgency','marketCommentary','relevance','sentimentClass','sentimentNegative','sentimentNeutral','sentimentPositive','sentimentWordRatio']
    # Filters out data with universe==0
    # X_train = X_train[X_train.universe==1]
    # y_train = y_train[y_train.universe==1]  
    # Save time for calculating score later. It is used to group and sum x_t values each day
    train_time = X_train['time']
    X_train = X_train[features_no_universe]
    y_train = y_train['returnsOpenNextMktres10']    
    # Filters out data with universe==0 for accurate scoring
    X_test = X_test[X_test.universe==1]
    y_test = y_test[y_test.universe==1]
    # Save time for calculating score later. It is used to group and sum x_t values each day
    test_time = X_test['time']
    X_test = X_test[features_no_universe]
    y_test = y_test['returnsOpenNextMktres10']   
    return X_train,X_test,y_train,y_test,train_time,test_time

# Draw graph of train vs eval scores. Visualize training process once it's done
def draw_train_eval_graph(evals_result,params):
    x_axix = range(1,len(evals_result['train']['sigma_score'])+1)
    train_sigma_score = evals_result['train']['sigma_score']
    eval_sigma_score = evals_result['eval']['sigma_score']

    plt.plot(x_axix,train_sigma_score,label='Train')
    plt.plot(x_axix,eval_sigma_score,label='Eval')
    plt.legend()
    print("eta: ",params['eta'],", max_depth: ",params['max_depth'])

# This will display real target vs predictions. Kind of a sanity check..
def compare_real_target_with_pred(x,y):
    # Compare predict and actual nextMKTres side by side
    input_for_pred = xgb.DMatrix(x.values)
    y_pred = bst.predict(input_for_pred,ntree_limit = bst.best_ntree_limit)
    data  = {'y_real':y.values, 'y_pred': y_pred}
    print(pd.DataFrame(data))
# sigma_score function is considered as a custom evaluation metric for xgboost
# example of how custom evaluation function is incorporated into xgboost's training can be found here : https://github.com/dmlc/xgboost/blob/master/demo/guide-python/custom_objective.py
def sigma_score(preds,dval):
    # get y_target values
    labels = dval.get_label()
    # call time parameter to be used for grouping, so that we can add x_t values for each day
    df_time = dval.params['extra_time']
    # instead of making any prediction above 0 as 1, I chose anything above the mean of predictions (I call it market average) to be 1
    preds[preds>preds.mean()]=1
    # anything between market average(prediction mean) and 0 were given 0. 
    preds[(preds<=preds.mean())&(preds>=0)]=0
    # any asset giving negative return...... -1 
    preds[preds<0]=-1
    # I assume you can take below approach too
    #preds[preds>0]=1
    #preds[preds<=0]=-1
    
    #calculate x_t and score as specified by the competition
    x_t = pd.Series(preds*labels)
    x_t_sum = x_t.groupby(df_time).sum()    
    score = (x_t_sum.mean())/(x_t_sum.std())
    return 'sigma_score', round(score,5)
import xgboost as xgb

# remember, this is not train_test_split from sklearn. This is my own function. It devides data with 30 days gap and doesn't allow the model to look into the future.
X_train,X_val,y_train,y_val,train_time,val_time=split_train_test_and_time(x,y,test_size=0.2)

# Define datasets that xgboost accepts
xgtrain = xgb.DMatrix(X_train.values,y_train.values)
xgval = xgb.DMatrix(X_val.values,y_val.values)

# We will 'inject' an extra parameter in order to have access to df_valid['time'] inside sigma_score without globals
xgtrain.params = {'extra_time': train_time.factorize()[0]}
xgval.params = {'extra_time': val_time.factorize()[0]}

# define parameters. I found learning rate of 0.3 and max_depth of 6 to be suitable.
params ={'eta':0.5, 'max_depth':5,'objective':'reg:linear','silent':1,'eval_metric':'rmse'}
# this allows cross validation. Make sure eval data is the latter one, so that the model will do an early stopping if eval data's sigma score doesn't increase.
# We want the training to stop when eval data's sigma score doesn't increase so that we don't overfit our model to the training data
evallist = [(xgtrain,'train'),(xgval,'eval')]
# Save evaluation metric scores for displaying later
evals_result = {}
# perform 400 rounds at maximum if early stopping doesn't happen
num_round = 400
# here, one thing to note is that our custom evaluation function 'sigma_score' is passed into 'feval'.
bst = xgb.train(params,xgtrain,num_round,evallist,evals_result=evals_result,feval=sigma_score,maximize=True,early_stopping_rounds=50,verbose_eval=10)
# # If early stopping is enabled during training, you can get predictions from the best iteration by using this-> y_test_pred = bst.predict(xgtest,ntree_limit = bst.best_ntree_limit)

# Compare predict and actual nextMKTres side by side
compare_real_target_with_pred(X_val,y_val)
# Compare predict and actual nextMKTres side by side
compare_real_target_with_pred(X_train,y_train)
draw_train_eval_graph(evals_result,params)
# Use functions defined previously to process data for the final submission
def make_my_confidence_predictions(market_obs_df,predictions_template_df):#, news_obs_df, predictions_template_df):
    x=pre_process_market_data_actual_competition(market_obs_df).copy()
    x = xgb.DMatrix(x.values)
    predictions_template_df.confidenceValue = bst.predict(x,ntree_limit = bst.best_ntree_limit)
    predictions_template_df.confidenceValue[predictions_template_df.confidenceValue>predictions_template_df.confidenceValue.mean()]=1
    predictions_template_df.confidenceValue[(predictions_template_df.confidenceValue<=predictions_template_df.confidenceValue.mean())&(predictions_template_df.confidenceValue>=0)]=0
    predictions_template_df.confidenceValue[predictions_template_df.confidenceValue<0]=-1
    return predictions_template_df

for (market_obs_df, news_obs_df, predictions_template_df) in env.get_prediction_days():    
    # Process news data
    news_obs_df=process_news_date(news_obs_df)
    # Truncate date so that both news and market can be combined
    news_obs_df=combined_index(news_obs_df).copy()
    # Truncate date so that both news and market can be combined
    market_obs_df=combined_index(market_obs_df).copy()
    market_obs_df=merge_market_news(market_obs_df,news_obs_df)
    market_obs_df=pre_process_market_data_actual_competition(market_obs_df)
    
    predictions_df = make_my_confidence_predictions(market_obs_df, predictions_template_df)
    env.predict(predictions_df)
print('Done!')
# Write submission file    
env.write_submission_file()

