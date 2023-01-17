from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()
market_train_df.returnsOpenPrevRaw1.min()
market_train_df[
    market_train_df.returnsOpenPrevRaw1 == market_train_df.returnsOpenPrevRaw1.min()
].T
features = ['time', 'open', 'returnsOpenPrevRaw10', 'returnsOpenPrevMktres10']
market_train_df[
    (market_train_df.assetCode == 'PBRa.N') & 
    (market_train_df.time >= '2007-05-01') &
    (market_train_df.time <= '2007-06-20')
][features]
market_train_df['raw_median'] = market_train_df.groupby('time').returnsOpenPrevRaw10.transform('median')
market_train_df['xy'] = market_train_df.returnsOpenPrevRaw10 * market_train_df.raw_median

roll = market_train_df.groupby('assetCode').rolling(window=20)

market_train_df['cov_xy'] = (
    (roll.xy.mean() - roll.returnsOpenPrevRaw10.mean() * roll.raw_median.mean()) * 20 / 19
).reset_index(0,drop=True)
market_train_df['var_y'] = roll.raw_median.var().reset_index(0,drop=True)
market_train_df['beta'] = (market_train_df['cov_xy'] / market_train_df['var_y'])
market_train_df['beta'] = market_train_df.groupby('assetCode')['beta'].shift(1)
market_train_df[
    (market_train_df.assetCode == 'PBRa.N') & 
    (market_train_df.time >= '2007-05-01') &
    (market_train_df.time <= '2007-06-20')
][features+['beta']]
