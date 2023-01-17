from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
news_train_df.head()
china_news = news_train_df[news_train_df['subjects'].str.contains('CN') | 
                           news_train_df['headline'].str.contains('China', 'Chinese')]
print(china_news.shape)
china_news.to_csv('china_news.csv', index = False)
indo_news = news_train_df[news_train_df['headline'].str.contains('Indonesia', 'Indonesian') | 
                          news_train_df['subjects'].str.contains('ID')]
print(indo_news.shape)
indo_news.to_csv('indo_news.csv', index = False)
not_china_news = china_news[china_news['subjects'].str.contains('CN') & 
                            ~(china_news['headline'].str.contains('China', 'Chinese'))]
print(not_china_news.shape)
not_china_news.head()
