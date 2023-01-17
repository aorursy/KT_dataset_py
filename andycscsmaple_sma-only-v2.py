import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
days = env.get_prediction_days()
all_asset_close = dict()
WINDOW_LONG = 30
WINDOW_SHORT = 5

for dayCount, (market_obs_df, news_obs_df, template_df) in enumerate(days):
    for index, row in market_obs_df.iterrows():
        asset = row['assetCode']
        if asset not in all_asset_close:
            all_asset_close[asset] = []
        else:
            all_asset_close[asset].append(row['close'])
            
        if len(all_asset_close[asset]) < WINDOW_LONG:
            template_df.loc[index, 'confidenceValue'] = 0.0
        else:
            sma_short = sum(all_asset_close[asset][-1*WINDOW_SHORT:]) / float(WINDOW_SHORT)
            sma_long = sum(all_asset_close[asset][-1*WINDOW_LONG:]) / float(WINDOW_LONG)
            template_df.loc[index, 'confidenceValue'] = 1.0 if sma_short > sma_long else -1.0
        
    env.predict(template_df)
    
    if dayCount % 50 == 0: 
        print('day ', dayCount+1)
 
env.write_submission_file()
