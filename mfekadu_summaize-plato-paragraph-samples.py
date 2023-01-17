!pip uninstall -y cupy-cuda101

!pip uninstall -y spacy
!pip install cupy-cuda101==6.5.0

!pip install spacy[cuda101]==2.2.4
!pip list | grep cupy

!echo "....."

!pip list | grep spacy

!echo "....."

# !pip install spacy[cuda]

!echo "....."

!pip list | grep spacy

!echo "....."

!pip list | grep cuda
# !pip uninstall -y cupy-cuda101

# !echo "....."

# !pip list | grep spacy

# !echo "....."

# !pip list | grep cuda

# !echo "....."

# !pip install cupy-cuda101
import cupy

import spacy
spacy.require_gpu()

# spacy.prefer_gpu()
import torch

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gpu_index = gpu.index if gpu is not "cpu" else -1 # -1 means CPU for transformers.Pipeline

gpu_index
from sklearn.utils import check_consistent_length, assert_all_finite

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import transformers



from transformers import pipeline
nlp = spacy.load("en_core_web_lg")



# test that nlp is working

nlp("apples").similarity(nlp("oranges"))

nlp(

    "\napple"  # newlines break cupy-cuda101==7.4.0

  ).similarity(

      nlp("banana")

  )
CSV_FILE = "/kaggle/input/stanford-plato-corpus/data_per_paragraph.csv"
df = pd.read_csv(CSV_FILE)
df.shape
df.head()
PARAGRAPH_COL = "section.paragraph.text"

PREAMBLE_COL = "preamble_text"

FILENAME_COL = "filename"
paragraph_series = df[PARAGRAPH_COL][df[PARAGRAPH_COL].notna()]

preamble_series = df[PREAMBLE_COL][df[PARAGRAPH_COL].notna()]



# raises error if bad data

check_consistent_length(preamble_series, paragraph_series)

assert_all_finite(preamble_series)

assert_all_finite(paragraph_series)
# manage memory

import gc

del df

gc.collect()
def _get_len(x):

    return len(x) if x is not np.nan else 0
length_series = (

    paragraph_series

    .apply(_get_len)

)



length_series.describe()
summarizer = pipeline(

    "summarization",

    device=gpu_index,

)
summarizer("Hello world.")
print("count\t:",  preamble_series.count())

print("unique\t:", len(preamble_series.unique()))
print("count\t:",  paragraph_series.count())

print("unique\t:", len(paragraph_series.unique()))
paragraph_series[paragraph_series.duplicated()].to_frame().head(10)
IDX = 2

sample_paragraph = paragraph_series[IDX]

sample_preamble = preamble_series[IDX]
print("-----"*10)

print("sample_paragraph".upper())

print(sample_paragraph)

print("-----"*10)

print("sample_preamble".upper())

print(sample_preamble)

print("-----"*10)
print("sample_paragraph is\t", len(sample_paragraph),"\tcharacters")
sample_summary_list = summarizer(

    sample_paragraph,

    max_length=512,

    min_length=100,

)
sample_summary = sample_summary_list[0]["summary_text"]
sample_summary_doc = nlp(sample_summary)

sample_preamble_doc = nlp(sample_preamble)

sample_paragraph_doc = nlp(sample_paragraph)
nlp("apple").similarity(nlp("apple"))
nlp("apple").similarity(nlp("orange"))
sample_summary_doc.similarity(

    sample_preamble_doc

)
sample_summary_doc.similarity(

    sample_paragraph_doc

)
print("-----"*10)

print("sample_paragraph".upper())

print(sample_paragraph)

print("-----"*10)

print("sample_summary_doc".upper())

print(sample_summary_doc)

print("-----"*10)

print("sample_preamble_doc".upper())

print(sample_preamble_doc)

print("-----"*10)
TEN_PERCENT = round(len(paragraph_series)*0.1)

ONE_PERCENT = round(len(paragraph_series)*0.01)

SAMPLE_N = ONE_PERCENT

print(SAMPLE_N)
_df = preamble_series.to_frame().join(paragraph_series)

sample_df = _df.sample(SAMPLE_N)

sample_df.head(3)
sample_df.to_csv("sample_df.csv")
i = 0

NUM_PARAGRAPHS = len(sample_df)
def _with_counter(fun, _N, _i=1):

    """

    print a progress count with each function call

    """

    global COUNTER_I, COUNTER_N

    # initialize for reuse within _fun

    COUNTER_I = _i

    COUNTER_N = _N

    def _fun(*args, **kwargs):

        global COUNTER_I, COUNTER_N

        print(

            f"{round(COUNTER_I/COUNTER_N, 3)}\t == \t{COUNTER_I} / {COUNTER_N}", 

            end="\r"

        )

        print("DONE",f"{COUNTER_I} / {COUNTER_N}","\t"*30) if COUNTER_I >= COUNTER_N else None

        COUNTER_I += 1

        return fun(*args, **kwargs)

    return _fun





# csum = _with_counter(sum, 5)

# for q in range(5):

#     csum([2,2])

# # only define with_counter when needed

# # else global scope problem

# csum2 = _with_counter(sum, 5)

# for q in range(5):    

#     csum2([2,2])





def summarize(

    text,

    smmry_key="summary_text", 

    # TODO: consider variable length based on text length

    max_length=512, 

    min_length=100,

):

    if text is not np.nan:

        smmary_list = summarizer(

            text,

            max_length=max_length, 

            min_length=min_length,

        )

        smmry = smmary_list[0][smmry_key]

        return smmry

    else:

        return ""
# csum = _with_counter(sum, 5)

# for q in range(5):

#     csum([2,2])

# # only define with_counter when needed

# # else global scope problem

# csum2 = _with_counter(sum, 5)

# for q in range(5):    

#     csum2([2,2])
csummarize = _with_counter(summarize, _N=SAMPLE_N)



sample_df["smmry"] = (

    sample_df[PARAGRAPH_COL]

    .apply(csummarize)

)
sample_df.head()
sample_df.to_csv("sample_df.csv")
def _get_sim(x):

    return nlp(x[0]).similarity(nlp(x[1]))
sample_df["par_sim_smmry"] = sample_df[[PARAGRAPH_COL,"smmry"]].apply(_get_sim, axis=1).astype(float)

sample_df["pre_sim_smmry"] = sample_df[[PREAMBLE_COL,"smmry"]].apply(_get_sim, axis=1).astype(float)

sample_df["pre_sim_par"] = sample_df[[PREAMBLE_COL,PARAGRAPH_COL]].apply(_get_sim, axis=1).astype(float)
sample_df.to_csv("sample_df.csv")
sample_df.describe()
!ls -lah
pd.read_csv("sample_df.csv").head()