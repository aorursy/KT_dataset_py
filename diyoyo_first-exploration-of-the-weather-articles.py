import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data=pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

data.head()



data[(data['Country/Region']=='France') & (data['ObservationDate']=='03/23/2020')]
article_meta=pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

article_meta.head()
article_meta[(pd.notna(article_meta['abstract'])) 

                  & (article_meta['abstract'].str.contains("weather"))

                  & (article_meta['abstract'].str.contains("correlation"))

                  & (article_meta['abstract'].str.contains("virus"))].filter(["abstract","pmcid", "pubmed_id"])

