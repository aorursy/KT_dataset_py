!pip install --upgrade transformers 

!pip install nb_black

%load_ext nb_black
import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from transformers import pipeline



# load BART summarizer

summarizer = pipeline(task="summarization")



# load the meta data from the CSV file using 3 columns (abstract, title, authors),

df = pd.read_csv(

    "/kaggle/input/CORD-19-research-challenge/metadata.csv",

    usecols=["title", "abstract", "authors", "doi", "publish_time"],

)

print(df.shape)

# drop duplicates

# df=df.drop_duplicates()

df = df.drop_duplicates(subset="abstract", keep="first")

# drop NANs

df = df.dropna()

# convert abstracts to lowercase

df["abstract"] = df["abstract"].str.lower()

# show 5 lines of the new dataframe

print(df.shape)

df.head()
import functools

from IPython.core.display import display, HTML

from nltk import PorterStemmer

from tqdm.notebook import tqdm_notebook



tqdm_notebook.pandas()



# list of lists for topic words realting to tasks

display(HTML("<h1>COVID-19 Risk Factors</h1>"))

display(

    HTML(

        "<h3>Table of Contents (ctrl f to search for the hash tagged words below to find that data table)</h3>"

    )

)

tasks = [

    ["comorbidities"],

    ["risk", "factors"],

    ["lung", "cancer"],

    ["hypertension"],

    ["heart", "disease"],

    ["chronic", "bronchitis"],

    ["cerebral", "infarction"],

    ["diabetes"],

    ["copd"],

    ["blood type", "type"],

    ["smoking"],

    ["effective", "reproductive", "number"],

    ["incubation", "period", "days"],

]





def summarize(row):

    summary = ""



    abstract = row["abstract"].split("abstract ")[-1]



    summary = summarizer(abstract, min_length=50, max_length=200)

    summary = summary[0]["summary_text"]



    if summary != "":

        authors = row["authors"].split(" ")

        link = row["doi"]

        title = row["title"]

        linka = "https://doi.org/" + link

        linkb = title

        summary = (

            '<p align="left">'

            + "<strong>Summary:</strong><br>"

            + summary

            + "<br><br>"

            + "<strong>Original:</strong><br>"

            + abstract

            + "</p>"

        )

        final_link = '<p align="left"><a href="{}">{}</a></p>'.format(linka, linkb)

        to_append = [

            row["publish_time"],

            authors[0] + " et al.",

            final_link,

            summary,

        ]

        df_length = len(df_table)

        df_table.loc[df_length] = to_append





# function to stem keywords into a common base word

def stem_words(words):

    stemmer = PorterStemmer()

    singles = []

    for w in words:

        singles.append(stemmer.stem(w))

    return singles





for z, search_words in enumerate(tqdm_notebook(tasks)):

    df_table = pd.DataFrame(columns=["pub_date", "authors", "title", "excerpt"])

    str1 = ""

    # a make a string of the search words to print readable search

    str1 = " ".join(search_words)

    search_words = stem_words(search_words)

    # add cov to focus the search the papers and avoid unrelated documents

    search_words.append("covid")

    # search the dataframe for all the keywords

    dfa = df[

        functools.reduce(

            lambda a, b: a & b, (df["abstract"].str.contains(s) for s in search_words)

        )

    ]

    search_words.pop()

    search_words.append("-cov-")

    dfb = df[

        functools.reduce(

            lambda a, b: a & b, (df["abstract"].str.contains(s) for s in search_words)

        )

    ]

    # remove the cov word for sentence level analysis

    search_words.pop()

    # combine frames with COVID and cov and drop dups

    frames = [dfa, dfb]

    df1 = pd.concat(frames)

    df1 = df1.drop_duplicates()

    df1 = df1.reset_index()



    display(HTML("<h3>Task Topic: " + str1 + "</h3>"))

    display(HTML("# " + str1 + " <a></a>"))



    print(df1.shape)

    # SUMMARIZATION for all abstracts

    #     df.progress_apply(summarize, axis=1, result_type="expand")

    # SUMMARIZATION for the first abstract in each category



    summarize(df1.loc[0])



    filename = str1 + ".csv"

    df_table.to_csv(filename, index=False)

    df_table = HTML(df_table.to_html(escape=False, index=False))

    display(df_table)