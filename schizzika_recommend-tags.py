# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import spacy
nlp = spacy.load('en')
doc1 = nlp('data science')

doc2 = nlp('artificial intelligence')
doc1.similarity(doc2)
economics_finance = nlp("interest rate loan debt mortgage lender payment bank income investor economy share stock tax asset capital")
for token1 in economics_finance:

    for token2 in economics_finance:

        print((token1.text, token2.text, "Similarity =>", token1.similarity(token2)))
economics_finance_list = [(token1.text, token2.text, token1.similarity(token2))

                          for token2 in economics_finance for token1 in economics_finance]
import pandas as pd

df = pd.DataFrame(economics_finance_list)
df.head()
df.columns = ['Token1', 'Token2', 'Similarity']
df.corr
df
#search = input("Type a tag: ")

search = 'loan'
new_df = df.loc[df['Token1'] == search]
new_df.sort_values(by = 'Similarity', ascending = False)
df_recommend = new_df[new_df['Similarity'] > 0.5]
df_recommend.sort_values(by = 'Similarity', ascending = False)