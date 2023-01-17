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



tnum = 5
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

    {"question": "Resources to support skilled nursing facilities and long term care"

                 " facilities."},

    {"question": "Mobilization of surge medical staff to address shortages in"

                 " overwhelmed communities."},

    {"question": "Age-adjusted mortality data for Acute Respiratory Distress"

                 " Syndrome (ARDS) with/without other organ failure – particularly"

                 " for viral etiologies."},

    {"question": "Extracorporeal membrane oxygenation (ECMO) outcomes data of"

                 " COVID-19 patients."},

    {"question": "Outcomes data for COVID-19 after mechanical ventilation adjusted"

                 " for age."},

    {"question": "Knowledge of the frequency, manifestations, and course of"

                 " extrapulmonary manifestations of COVID-19, including, but not"

                 " limited to, possible cardiomyopathy and cardiac arrest."},

    {"question": "Application of regulatory standards (e.g., EUA, CLIA) and ability"

                 " to adapt care to crisis standards of care level."},

    {"question": "Approaches for encouraging and facilitating the production of"

                 " elastomeric respirators, which can save thousands of N95"

                 " masks."},

    {"question": "Best telemedicine practices, barriers and faciitators, and"

                 " specific actions to remove/expand them within and across state"

                 " boundaries."},

    {"question": "Guidance on the simple things people can do at home to take care"

                 " of sick people and manage disease."},

    {"question": "Oral medications that might potentially work."},

    {"question": "Use of AI in real-time health care delivery to evaluate"

                 " interventions, risk factors, and outcomes in a way that could"

                 " not be done manually."},

    {"question": "Best practices and critical challenges and innovative solutions"

                 " and technologies in hospital flow and organization, workforce"

                 " protection, workforce allocation, community-based support"

                 " resources, payment, and supply chain management to enhance"

                 " capacity, efficiency, and outcomes."},

    {"question": "Efforts to define the natural history of disease to inform"

                 " clinical care, public health interventions, infection"

                 " prevention control, transmission, and clinical trials."},

    {"question": "Efforts to develop a core clinical outcome set to maximize"

                 " usability of data across a range of trials."},

    {"question": "Efforts to determine adjunctive and supportive interventions"

                 " that can improve the clinical outcomes of infected patients"

                 " (e.g. steroids, high flow oxygen)."}

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