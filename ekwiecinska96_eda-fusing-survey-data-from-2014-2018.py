import numpy as np 

import pandas as pd

import plotly.graph_objs as go

import spacy

import re

from IPython.display import display, HTML

import plotly.plotly as py

import plotly.offline as pyo

import plotly.graph_objs as go

import os

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode(connected=True)
data_paths = {

    '2014': '../input/mental-health-in-techology-survey-2014-and-2016/survey_2014.csv',

    '2016': '../input/mental-health-in-techology-survey-2014-and-2016/survey_2016.csv',

    '2017': '../input/osmi-mental-health-in-tech-survey-2017/OSMI Mental Health in Tech Survey 2017.csv',

    '2018': '../input/osmi-mental-health-in-tech-survey-2018/OSMI Mental Health in Tech Survey 2018.csv'

}



raw_data = {

    date: pd.read_csv(path, index_col=False) for date, path in data_paths.items()

}
def clean_columns(dataframe, year):



    dataframe.columns = map(str.lower, dataframe.columns)



    # Remove HTML artifacts

    dataframe.rename(columns=lambda colname: re.sub('</\w+>', '', colname), inplace=True)

    dataframe.rename(columns=lambda colname: re.sub('<\w+>', '', colname), inplace=True)



    # Standardise demographic questions

    dataframe.rename(columns={'what is your age?': 'age', 'what is your gender?': 'gender',

                              'what is your race?': 'race'},

                     inplace=True)



    # Following the 2014 convention where 'country' refers to country of living

    unused_columns = ['country_work', 'state_work', 'timestamp']

    for column in unused_columns:

        if column in dataframe.columns:

            dataframe.drop(columns=column, inplace=True)



    dataframe['year'] = year



    if {'#', 'start date (utc)', 'submit date (utc)', 'network id'}.issubset(set(dataframe.columns)):

        dataframe.drop(columns=['#', 'start date (utc)', 'submit date (utc)', 'network id'], inplace=True)



    # Drop duplicated columns

    dataframe = dataframe.loc[:, ~dataframe.columns.duplicated()]



    dataframe.reset_index(inplace=True, drop=True)



    return dataframe

for dataset in ['2014', '2016', '2017', '2018']:

    raw_data[dataset] = clean_columns(raw_data[dataset], dataset)

    

initial_concat = pd.concat([raw_data['2014'], raw_data['2016']], ignore_index=True, sort=True)

interm_concat = pd.concat([raw_data['2017'], raw_data['2018']], ignore_index=True, sort=True)

survey_dataframe = pd.concat([initial_concat, interm_concat], ignore_index=True, sort=True)



del initial_concat, interm_concat

display(HTML(survey_dataframe.head(3).to_html()))
def generate_missing_value_heatmap(dataframe):

    """

    Generates a plotly heatmap to graphically display missing values in a pandas dataframe



    :param dataframe: Pandas dataframe, missing values should be of type `numpy.nan`

    :type dataframe: Pandas DataFrame



    :return:

    :rtype: Python dictionary with keys 'data' and 'layout', containing a plotly.Heatmap and plotly.Layout object.



    """

    val_array = np.array(dataframe.fillna(-99).values)

    val_array[np.where(val_array != -99)] = 0

    val_array[np.where(val_array == -99)] = 1



    data = [

        go.Heatmap(

            z=val_array,

            x=dataframe.columns,

            y=dataframe.index,

            colorscale='Reds',

           hovertemplate='Question: %{x}\n Missing?: %{z}'

        )

    ]



    layout = go.Layout(

    title=dict(text="Missing data heatmap (red values indicate missing). Hover to see responses with missing values.",

               font=dict(size=24)),

    autosize = True, 

    xaxis=dict(

        showticklabels=False, 

        ticks="", 

        showgrid=False,

        zeroline=False,

        automargin=False,

        tickmode='array',

    ),

    yaxis=dict(

        autorange=True,

        tickmode='array',

        showgrid=False,

        zeroline=False,

        showline=False,

        ticks="",

        automargin=False,

        showticklabels=False),

    )



    fig = dict(data=data, layout=layout)



    return fig
iplot(generate_missing_value_heatmap(survey_dataframe))
# !python -m spacy download en_core_web_md



# word_vecs = spacy.load('en_core_web_md')
def compute_similarity_matrix(documents, word_vecs):

    if not isinstance(documents, list):

        raise ValueError("Documents must be a list of strings")

    else:

        similarity_matrix = np.zeros(shape=(len(documents), len(documents)))



        for question_i in range(len(documents)):

            for question_j in range(len(documents)):

                if question_i == question_j:

                    continue

                else:

                    question_i_vec = word_vecs(documents[question_i])

                    question_j_vec = word_vecs(documents[question_j])

                    similarity_matrix[question_i, question_j] = question_i_vec.similarity(question_j_vec)

        return similarity_matrix

    

def remove_punctuation(documents):

    punctuation = re.compile('[\?\(\)\.\,]')

    if isinstance(documents, list):

        clean_documents = [''] * len(documents)

        for i, document in enumerate(documents):

            clean_documents[i] = re.sub(punctuation, '', document)

    elif isinstance(documents, str):

        clean_documents = re.sub(punctuation, '', documents)

    else:

        raise ValueError("Documents must be a list of strings or a string")



    return clean_documents
header = remove_punctuation(list(survey_dataframe.columns))

# This is a costly operation, and needs a SpaCy model with word vectors to be performed.

# similarity_matrix = compute_similarity_matrix(header, word_vecs)



# In kernel mode, let us just load this from the pre-computed weights

similarity_matrix = np.load('../input/mental-health-in-techology-survey-2014-and-2016/similarity_matrix.npy')
def generate_similarity_heatmap(similarity_matrix, labels=None):

    

    if labels is None or not isinstance(labels, list):

        labels = np.arange(similarity_matrix.shape[0])

    elif similarity_matrix.shape[0] != similarity_matrix.shape[1]:

        raise ValueError("Please provide a square matrix")

    else:

        pass

    

    data = go.Heatmap(

            z=similarity_matrix,

            x=labels,

            y=labels,

            colorscale='Hot',

        )



    layout = go.Layout(

        title=dict(text="Zoomable Similarity Matrix (hover to see the similarity between each label)",

                   font=dict(size=24)),

        autosize = True, 

        xaxis=dict(

            showticklabels=False, 

            ticks="", 

            showgrid=False,

            zeroline=False,

            automargin=False,

            tickmode='array',

        ),

        yaxis=dict(

            autorange=True,

            tickmode='array',

            showgrid=False,

            zeroline=False,

            showline=False,

            ticks="",

            automargin=False,

            showticklabels=False),

    )



    fig = dict(data=[data], layout=layout)

    return fig
iplot(generate_similarity_heatmap(similarity_matrix, labels=header))