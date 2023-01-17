# SpaCy setup.

import spacy

spacy.prefer_gpu()

nlp = spacy.load("en_core_web_lg")

nlp.max_length = 2e6



# Compare two documents

doc1 = nlp("I like fast food")

doc2 = nlp("I like pizza")

similarity = doc1.similarity(doc2)



similarity
!spacy info
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import spacy

from spacy.matcher import PhraseMatcher

from IPython.core.display import display, HTML

import re

import matplotlib.pyplot as plt

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



tnum = 4
data = pd.read_json("../input/covid19-literature/df_covid_3.json")
display(HTML(f"""<h2>Data description</h2><br/>

<ul>

    <li>

        The dataset contains metadata and content from 

        <b>{data.shape[0]:,} scientific research papers</b>.

    </li>

    <li>

        <b>Each record contains the following variables:

        </b> {", ".join(data.columns.values)}

    </li>

    <li>

        The <code>body_text</code> feature contains the 

        sceintific content and ranges in size from 

        <b>{data.body_word_count.min():,} to 

        {data.body_word_count.max():,} words</b> with an

        <b>average of {round(data.body_word_count.mean()):,}

        words</b>.

    </li>

</ul>"""))



display(HTML("<p><b>Here is a random sample of"

             " four records from the dataset:</b></p>"))

data.sample(4)
questions = [

    {"question": "Effectiveness of drugs being developed and tried to treat COVID-19"

                 " patients."},

    {"question": "Effectiveness of drugs being developed and tried to treat COVID-19"

                 " patients; Clinical and bench trials to investigate less common"

                 " viral inhibitors against COVID-19 such as naproxen,"

                 " clarithromycin, and minocyclinethat that may exert effects on"

                 " viral replication."},

    {"question": "Methods evaluating potential complication of Antibody-Dependent"

                 " Enhancement (ADE) in vaccine recipients."},

    {"question": "Exploration of use of best animal models and their predictive"

                 " value for a human vaccine."},

    {"question": "Capabilities to discover a therapeutic (not vaccine) for the"

                 " disease, and clinical effectiveness studies to discover"

                 " therapeutics, to include antiviral agents."},

    {"question": "Alternative models to aid decision makers in determining how to"

                 " prioritize and distribute scarce, newly proven therapeutics as"

                 " production ramps up. This could include identifying approaches"

                 " for expanding production capacity to ensure equitable and"

                 " timely distribution to populations in need."},

    {"question": "Efforts targeted at a universal coronavirus vaccine."},

    {"question": "Efforts to develop animal models and standardize challenge"

                 " studies."},

    {"question": "Efforts to develop prophylaxis clinical studies and prioritize"

                 " in healthcare workers."},

    {"question": "Approaches to evaluate risk for enhanced disease after"

                 " vaccination."},

    {"question": "Assays to evaluate vaccine immune response and process"

                 " development for vaccines, alongside suitable animal models"

                 " [in conjunction with therapeutics]."}

]
display(HTML("<h3>Questions and their keywords</h3>"))



for i, q in enumerate(questions):

    qdoc = nlp(q["question"])

    key_tokens = [token.text.lower() for token in qdoc

                  if token.pos_ not in ["PUNCT","SYM","PART","ADV"]

                  and not token.is_stop

                  and len(token.text) > 2]

        

    questions[i]["keywords"] = key_tokens

    keywords = ", ".join(key_tokens)

    display(HTML(f"""<p><strong>Question #{i+1}:</strong> {qdoc.text}<br/>

                        <strong>Keywords:</strong> {keywords}

                     </p>"""))
def draw_histograms(df, variables, n_rows, n_cols):

    """Plots histograms

    :param df: pandas dataframe

    :param variables: columns to plot

    :param n_rows: number of rows

    :param n_cols: number of columns

    :see: https://stackoverflow.com/questions/29530355/plotting-multiple-histograms-in-grid#answer-29530596

    """

    fig=plt.figure(figsize=(30,n_rows*8))

    for i, var_name in enumerate(variables):

        ax=fig.add_subplot(n_rows,n_cols,i+1)

        df[var_name].hist(bins=100,ax=ax)

        ax.set_title(

            f"Distribution of question #{i+1} similarity scores"

            , fontdict={"fontsize":34}

        )

    plt.show()



draw_histograms(

    data,

    [f"t{tnum}_q{i}_similarity" for i, q in enumerate(questions)]

    , math.ceil(len(questions)/2)

    , 2

)
plt.figure(figsize=(30,20))

plt.title(

    "Quantile-based distributions of each question's similarity scores"

    , fontdict={"fontsize":34}

)

data.boxplot(column=[f"t{tnum}_q{i}_similarity" for i, q in enumerate(questions)])
# Adjust this variable as desired to increase

# or decrease the number of results displayed.

num_results = 3



# Generate table of contents.

display(HTML('<h2 id="toc">Table of Contents</h2>'))

for i, q in enumerate(questions):

    display(HTML(

        f'<b>{i+1}</b>. <a href="#q{i}">{q["question"]}</a><br/>'

    ))

    

# Generate the display for the top results per question.

for i, q in enumerate(questions):

    sim_id = f"t{tnum}_q{i}_similarity"

    

    # Create the PhraseMatcher object. The tokenizer is the first argument.

    # Use attr = 'LOWER' to make consistent capitalization

    matcher = PhraseMatcher(nlp.vocab, attr='LEMMA')

    patterns = [

        nlp(term) for term in q["keywords"]

        + ["covid-19","covid19","coronavirus","corona"]

    ]

    matcher.add("MENU",            # Just a name for the set of rules we're matching to

            None,              # Special actions to take on matched words

            *patterns)

    

    display(HTML(

        f"""<h1 id="q{i}">{q["question"]}</h1>

        <a href="#toc" title="Table of Contents">Back to top ↑</a>"""

    ))



    

    dat = data.sort_values(by=f"t{tnum}_q{i}_similarity", ascending=False)

    

    for r in dat[:num_results].iterrows():

        i = r[0]

        row = r[1]

        

        doc = nlp(row["body_text"])

        matches = matcher(doc)        



        if matches:

            excerpt = row["body_text"]

            

            for m in matches:

                match = doc[m[1]:m[2]].text

                excerpt = re.sub(

                    f'([ -])(?!<mark style="background:yellow">){match}'

                    '(?!</mark>)([ ,.?!-])'

                    , f'\\1<mark style="background:yellow">{match}</mark>\\2'

                    , excerpt)

        else:

            if row["body_text"]:

                excerpt = row["body_text"]

            elif row["abstract"]:

                excerpt = row["abstract"]

            else:

                excerpt = "NO TEXT FOUND!"

                

        excerpt = re.sub(

            ' (?!<mark style="background:lightblue">)([0-9.-−]+)?'

            ' (days?|hours?|weeks?|months?|years?)([ ,.?!])'

            , ' <mark style="background:lightblue">\\1 \\2</mark>\\3'

            , excerpt)

        

        excerpt = re.sub(

            '(?!<mark style="background:lightgreen">)([0-9.-]+) ?(%)'

            , '<mark style="background:lightgreen">\\1 \\2</mark>'

            , excerpt)

        

        display(HTML(f"""<p>

                <strong>Title:</strong> {row["title"]}<br/>

                <strong>Authors:</strong> {row["authors"]}<br/>

                <strong>Journal:</strong> {row["journal"]}<br/>

                <strong>ID:</strong> {row["paper_id"]}<br/>

                <strong>Similarity score:</strong> {row[f"{sim_id}"]}<br/>

                <strong>Number of keyword matches:</strong> {len(matches)}<br/>

                <strong>Question:</strong> {q["question"]}<br/>

                <a href="#toc" title="Table of Contents">Back to top ↑</a>

            </p>

            <blockquote>{excerpt}</blockquote>"""))