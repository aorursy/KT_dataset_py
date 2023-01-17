# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import gc



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
CSV_FILE = "/kaggle/input/stanford-plato-corpus/simple_data.csv"
df = pd.read_csv(CSV_FILE)
df.size
df.shape
df.head()
df.columns
def string_to_python_list(x):

    return eval(x) if isinstance(x,str) else None



df['related_entries_list'] = df['related_entries_list'].apply(string_to_python_list)
df['sections'] = df['sections'].apply(string_to_python_list)
df.dtypes
example_df = pd.DataFrame({'A': [[1, 2, 3], 'foo', [], [3, 4]], 'B': 1})

example_df
example_df.explode("A")
# related_topic_per_row_df = df.explode("related_entries_list")

# related_topic_per_row_df.head()
# section_per_row_and_related_topic_per_row_df = related_topic_per_row_df.explode("sections")
# section_per_row_and_related_topic_per_row_df.head()
df = df.explode("sections")
df.columns
# df = df.rename(columns={"related_entries_list": "related_topic"})

df = df.rename(columns={"sections": "section"})

df = df.rename(columns={"filenames": "filename"})
df.head()
df['section'].values[1]
df['section'].values[2].keys()
df['section'].values[2]['paragraphs'][0].keys()
def safe_get(getter=lambda x, n: x):

    def _get(x):

        if x == None:

            return None

        elif isinstance(x, dict):

            return getter(x)

        else:

            raise NotImplementedError()

    return _get



def _id_getter(x):

    return x["id"]



def _ht_getter(x):

    return x["heading_text"]



def _p_getter(x):

    return x["paragraphs"]



def _text_getter(x):

    return x["text"]
df['section.id'] = df['section'].apply(safe_get(_id_getter))

df['section.heading_text'] = df['section'].apply(safe_get(_ht_getter))

df['section.paragraphs'] = df['section'].apply(safe_get(_p_getter))
df = df.explode("section.paragraphs")
df = df.rename(columns={"section.paragraphs": "section.paragraph"})
df['section.paragraph.id'] = df['section.paragraph'].apply(safe_get(_id_getter))

df['section.paragraph.text'] = df['section.paragraph'].apply(safe_get(_text_getter))
df.head()
df.columns
mask = [

    'filename', 'filetype', 'topic', 'title', 'author', 'creator',

    'preamble_text', 

    #'section', 

    'related_entries_list', 

    # 'plain_text',

    'section.id', 'section.heading_text', 

    # 'section.paragraph', 

    'section.paragraph.id', 'section.paragraph.text'

]

df = df[mask]
df.size
df.shape
len(df)
df.head()
def chunks(lst, N):

    """Yield N successive `len(lst)//N`-sized chunks from lst."""

    chunk_size = len(lst)//N

    for i in range(0, len(lst), len(lst)//N):

        yield lst[i:i + chunk_size]
TOTAL_CHUNKS = 10000
df_generator = chunks(df, TOTAL_CHUNKS)
CSV_FILENAME = "data_per_paragraph.csv"



# save first one

first_with_header = next(df_generator)



first_with_header.to_csv(CSV_FILENAME, mode='w', header=True)
del first_with_header

gc.collect()
i = 2

COLLECTED = gc.collect()

print(COLLECTED)

for x in df_generator:

#     if (i % 1000 == 0):

#         COLLECTED = gc.collect()

    print(f"appending... {i} / {TOTAL_CHUNKS}\r", end="")

    x.to_csv(CSV_FILENAME, mode='a', header=False)

    i+=1
pd.read_csv(CSV_FILENAME).head()
df[df["section.paragraph.text"].isna()].groupby("filetype").count()
na_par_df = df[df["section.paragraph.text"].isna()]

(

    na_par_df.join(

        na_par_df["filename"].apply(lambda x: os.path.split(x)[1]),

        rsuffix="_____"

    ).groupby("filename_____")

    .count()

    .sort_values(["filename","filetype","topic","title","author"], ascending=False)

)