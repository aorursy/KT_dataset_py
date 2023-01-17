from kaggle.competitions import twosigmanews

# Read data but just once
if 'env' not in globals():
    env = twosigmanews.make_env()
    (market_df, news_df) = env.get_training_data()
market_df.to_pickle('market_df.pkl')
news_df.to_pickle('news_df.pkl')
