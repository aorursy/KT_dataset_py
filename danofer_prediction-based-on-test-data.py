from kaggle.competitions import twosigmanews
import numpy as np
import pandas as pd

env = twosigmanews.make_env()
def predict(df):
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
market_obs_df = None
pred_df = None

for (m_df, n_df, predictions_template_df) in env.get_prediction_days():
    env.predict(predictions_template_df)
    predictions_template_df['time'] = m_df.time.min()
    if market_obs_df is None:
        market_obs_df = m_df
        pred_df = predictions_template_df
    else:
        market_obs_df = market_obs_df.append(m_df, ignore_index=True)
        pred_df = pred_df.append(predictions_template_df, ignore_index=True)

pred_df.assetCode = pred_df.assetCode.astype(str)
pred_df.time = pd.to_datetime(pred_df.time)
t = predict(market_obs_df)

# Score estimation
t['returns'] = t.confidenceValue * t.predictMktres10
day_returns = t.groupby('time').returns.sum()
day_returns.mean() / day_returns.std()
pred_df.drop(columns=['confidenceValue'], inplace=True)
pred_df = pred_df.merge(t[['time', 'assetCode', 'confidenceValue']], how='left', on=['time', 'assetCode'])\
.fillna(0)
pred_df.time = pred_df.time.dt.date
pred_df[['time', 'assetCode', 'confidenceValue']].to_csv('submission.csv', index=False, float_format='%.8f')
