import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



pd.set_option('display.max_rows', 100)
df_reviews = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')



df_reviews['Reviews'] = pd.to_numeric(df_reviews['Reviews'],errors='coerce')

df_reviews['Installs'] = pd.to_numeric(df_reviews['Installs'],errors='coerce')
df_reviews[['App','Genres','Rating','Reviews']][df_reviews['Reviews']>1000].sort_values(by=['Rating','Reviews'],ascending=[False,False])[:20]