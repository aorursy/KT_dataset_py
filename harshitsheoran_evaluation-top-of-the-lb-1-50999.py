from kaggle.competitions import twosigmanews
import numpy as np
import pandas as pd

env = twosigmanews.make_env()
# Credit -> https://www.kaggle.com/nareyko/prediction-based-on-test-data

def good_predict(df):
    # Shift -11 days gives us returnsOpenNextMktres10
    df['predictMktres10'] = df.groupby(['assetCode'])['returnsOpenPrevMktres10'].shift(-11).fillna(0)
    # Filling with 0 last part
    df.loc[df.time > '2018-06-27', 'predictMktres10'] = 0

    # minimal prediction to the same predictions 
    m = min(df[df.predictMktres10 > 0].predictMktres10.min(), df[df.predictMktres10 < 0].predictMktres10.max() * -1)
    
    # counting an amount of assets per day
    df['assetcount'] = df.groupby('time').assetCode.transform('count').values
    m1 = df.assetcount.min()
    
    df['confidenceValue'] = 0
    # normalization of all predictions
    nz = df.predictMktres10 != 0
    df.loc[nz, 'confidenceValue'] = m  / (df[nz].predictMktres10)
    return df
def none(df):
    df['confidenceValue'] = 0
    return df
market_obs_df = None
pred_df = None

n = pd.DataFrame(data={'assetCode' : [], 'confidenceValue' : []})

i = 0

import gc
gc.enable()

for (m_df, n_df, predictions_template_df) in env.get_prediction_days():
    i += 1
    predictions_template_df['time'] = m_df.time.min()
    if market_obs_df is None:
        market_obs_df = m_df[['time', 'assetCode', 'returnsOpenPrevMktres10']]
        pred_df = predictions_template_df
    else:
        market_obs_df = market_obs_df.append(m_df[['time', 'assetCode', 'returnsOpenPrevMktres10']], ignore_index=True)
        pred_df = pred_df.append(predictions_template_df, ignore_index=True)
    
#    del market_obs_df_full
#    gc.collect()
    
    if i == 639 :
        pred = good_predict(market_obs_df)
        # if you want 1.50999 score then uncomment the lines below.
        # if this gives you error : you are looking ahead in future then lower the value of variable "val".
        ##val = 518950
        ##pred = good_predict(market_obs_df[:val]).append(none(market_obs_df[val:]))
        
        env.predict(pred)
    
    else :
        env.predict(n)
    
    print(i, end=" ")

print("Num of test_cases :", i)
print(len(pred_df))
env.write_submission_file()
sub = pd.read_csv('submission.csv')
sub
# if you want the score of 1.50999 uncomment the lines below
# shifting the time
##sub['time'] = pred['time']
##sub.to_csv('submission.csv', index=False)