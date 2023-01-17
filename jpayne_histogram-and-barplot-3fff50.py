# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import math
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


from wordcloud import WordCloud, STOPWORDS

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input"]).decode("utf8").strip()
# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)
for size in range(1,7):
    length = 10 ** size
    with open("../input/executed_offenders_with_additional_info.csv", 'rb') as rawdata:
        result = chardet.detect(rawdata.read(length))

    # check what the character encoding might be
    print(size, length, result)
df = pd.read_csv('../input/executed_offenders_with_additional_info.csv', encoding = 'Windows-1252')
df.head()

df_age = df.loc[df['AgeTimeOfOffense'].notnull()]
df_age['TimeOnDeathRow'] = df_age['Age'] - df_age['AgeTimeOfOffense']
df_age.head(10)
df_age.groupby(['Race'], as_index=False).mean()
stripper = lambda x: x.strip()

df_age['Race'] = df_age['Race'].apply(stripper)
df_age.groupby('Race', as_index=False).mean()
df['Race'] = df['Race'].apply(stripper)
df.groupby('Race', as_index=False).mean()
len_state = lambda x: len(x)
df['lengthStatement'] = df['LastStatement'].apply(len_state)
df.head()
dec_state = lambda x: 1 if 'This offender' in x else 0
df['declined'] = df['LastStatement'].apply(dec_state)
df.head()
df.groupby('Race', as_index=False).mean()
df_len_statement = df[['Age','lengthStatement']]
df_len_statement.head()
df_len_statement.plot.scatter(x='Age', y='lengthStatement')
df_len_statement['Age'].corr(df['lengthStatement'])

df_len_statement = df_len_statement.loc[(df_len_statement['lengthStatement'] < 2000)]
df_len_statement.plot.scatter(x='Age', y='lengthStatement')
last_words = ''
for index, row in df.iterrows():
    last_words += row['LastStatement']
last_words
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(last_words)
#remove 'declined to make a statement'
remove_declined = df.loc[(df['declined'] == 0)]
show_wordcloud(remove_declined['LastStatement'])