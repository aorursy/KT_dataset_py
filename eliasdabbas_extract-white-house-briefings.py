import os



import pandas as pd

pd.options.display.max_columns = None
os.listdir('/kaggle/input/the-white-house-website/')
crawl_df = pd.concat([pd.read_csv('/kaggle/input/the-white-house-website/' + file)

                      for file in os.listdir('/kaggle/input/the-white-house-website/') if 'wh_crawl' in file],

                     ignore_index=True)

crawl_df.head(2)
briefings_df = crawl_df[crawl_df['url'].str.contains('briefings-statements/.{3,}')].filter(regex='briefing|^url$').copy()

briefings_df['briefing_date'] = pd.to_datetime(briefings_df['briefing_date'].str.split('@@').str[0])

briefings_df['briefing_category'] = briefings_df['briefing_category'].str.split('@@').str[0].str.strip()

briefings_df['briefing_body_text'] = briefings_df['briefing_body_text'].str.replace('@@', '\n')

briefings_df['full_body_text'] = [value.strip() if pd.notna(value) else crawl_df['body_text'][index]

                                  for index, value in briefings_df['briefing_body_text'].iteritems()]





briefings_df
na_index = briefings_df[briefings_df['briefing_body_text'].isna()].index

na_index
(crawl_df.loc[na_index, 'body_text'] == briefings_df['full_body_text'][na_index]).all()
briefings_df['briefing_body_text'] = briefings_df['full_body_text']

briefings_df = briefings_df.drop('full_body_text', axis=1)

briefings_df.sample(5)
briefings_df[briefings_df['briefing_body_text'].eq('')]['url'].tolist()
briefings_df.set_index('briefing_date').resample('A')['url'].count()
briefings_df.set_index('briefing_date').resample('M')['url'].count()
briefings_df['briefing_category'].value_counts(dropna=False).reset_index().style.background_gradient('cividis')
briefings_df
briefings_df.info()
print(briefings_df['briefing_body_text'].iloc[204])