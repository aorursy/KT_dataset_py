#import pandas as pd
import numpy as np
from kaggle.competitions import twosigmanews
#from sklearn.naive_bayes import GaussianNB
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
market_train_df.dropna(inplace = True)
market_train_df.reset_index(inplace = True)
df_market_1 = market_train_df[['time','assetName','open','close','volume']]
df_news_1 = news_train_df[['time','assetName','sentimentNegative','sentimentPositive']]
df_market_1['time_new'] = df_market_1['time'].dt.floor('d')
df_news_1['time_new'] = df_news_1['time'].dt.floor('d')
df_news_1.drop(['time'], inplace = True, axis = 1)
df_market_1.drop(['time'], inplace = True, axis =1 )
df_market_apple = df_market_1[df_market_1['assetName'] == 'Apple Inc']
df_news_apple = df_news_1[df_news_1['assetName'] == 'Apple Inc']
df_market_apple.head()
df_news_apple.head()
days = env.get_prediction_days()
#(market_obs_df, news_obs_df, predictions_template_df) = next(days)
next(days)
def make_random_predictions(predictions_df):
    predictions_df.confidenceValue = 2.0 * np.random.rand(len(predictions_df)) - 1.0
make_random_predictions(predictions_template_df)
env.predict(predictions_template_df)
(market_obs_df, news_obs_df, predictions_template_df) = next(days)
make_random_predictions(predictions_template_df)
env.predict(predictions_template_df)
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    make_random_predictions(predictions_template_df)
    env.predict(predictions_template_df)
print('Done!')
env.write_submission_file()
import os
print([filename for filename in os.listdir('.') if '.csv' in filename])
import pandas as pd
submissions = pd.read_csv('./submissions.csv')
submissions.head()
