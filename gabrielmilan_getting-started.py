import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_news = pd.read_csv('/kaggle/input/multipurpose-world-news-dataset/news.csv')
df_news.count()
samples = df_news.sample(n=5)



for idx in samples.index:

    sample = samples.loc[idx]

    print ("===> {}".format(sample['title']))

    if sample['description'] != 'nan':

        print ("   - {}".format(sample['description']))

    print ("   - Published at {} on {}".format(sample['source'], sample['timestamp']))

    print ("\n"*3)