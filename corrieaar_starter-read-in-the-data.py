import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import ast
df = pd.read_csv("/kaggle/input/data/data.csv", index_col=0, 

                 converters={"keyword_name": ast.literal_eval, # these are list columns, need to use ast to actually get lists and not strings

                            "keyword_id": ast.literal_eval,

                            "country_id": ast.literal_eval,

                            "country_name": ast.literal_eval,

                            "has_parts": ast.literal_eval})



# make merging later easier

df["creative_work_id"] = df["creative_work_id"].str.replace("/", "", n=1).str.replace("/", "_")

df.head(3)
df.shape
df[df.type == "http://schema.org/NewsArticle"].shape
articles_text = [text_file for text_file in os.listdir("/kaggle/input/data/texts") if text_file.endswith(".txt")]



texts = []

for path in articles_text:

    with open(os.path.join("/kaggle/input/data/texts", path), "r") as file:

        texts.append(file.read())



data = pd.DataFrame().from_dict({"path": articles_text, "text": texts})



data["creative_work_id"] = data["path"].str.replace(".txt", "")



# merge the texts with the rest of the data

data.merge(df, on="creative_work_id", how="left").head(3)
articles_html = [text_file for text_file in os.listdir("/kaggle/input/data/html") if text_file.endswith(".html")]



html = []

for path in articles_html:

    with open(os.path.join("/kaggle/input/data/html", path), "r") as file:

        html.append(file.read())



html[0][0:500]
# !pip install newspaper3k 
from newspaper import Article

from newspaper.outputformatters import OutputFormatter

from newspaper.configuration import Configuration



conf = Configuration()

conf.keep_article_html = True
art = Article('', keep_article_html=True)

art.set_html(html[0])

art.parse()



art.nlp()
art.keywords
art.summary
import re

re.findall(r"(?:<a .*?href=\")(.*?)(?:\")", art.html)[0:12]