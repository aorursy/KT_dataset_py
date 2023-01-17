import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # regex
DATA_PATH = "/kaggle/input/bangla-news-datasets-from-bdpratidin/1000DaysNews.csv"
all_news = pd.read_csv(DATA_PATH)
all_news.info()
punc = '[‘’,"":\']'
articles = all_news["article"].agg(lambda sent: re.sub(punc, '', str(sent).strip()))
sentences = []

for article in articles:
    sentences += [sentence.strip() for sentence in re.compile('[।?!\n]').split(article) if len(sentence) != 0 and len(sentence.split()) > 2]
    
sentences += [headline for headline in all_news["title"].agg(lambda sent: re.sub(punc, '', str(sent).strip())) if len(headline.split()) > 2]
sent_df = pd.DataFrame(sentences, columns=["sentence"])
sent_df["length"] = [len(sent) for sent in sent_df["sentence"]]
sent_df["word_count"] = [len(sent.split()) for sent in sent_df["sentence"]]
sent_df = sent_df.drop_duplicates("sentence")
sentences = sent_df["sentence"].tolist()
print(sent_df.describe().astype('int64'))
print(f"So, there are {len(sentences)} sentences. The minimum and maximum length of those sentences is 7 and 12839. And the min and max word count of those sentences is 3 and 1890.")
