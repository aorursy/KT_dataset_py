from qa_functions import *
import numpy as np 

import pandas as pd



# keep only docsuments with covid -cov-2 and cov2

def search_focus(df):

    dfa = df[df['abstract'].str.contains('covid')]

    dfb = df[df['abstract'].str.contains('-cov-2')]

    dfc = df[df['abstract'].str.contains('cov2')]

    frames=[dfa,dfb,dfc]

    df = pd.concat(frames)

    df=df.drop_duplicates(subset='title', keep="first")

    return df



# load the meta data from the CSV file using 3 columns (abstract, title, authors),

df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv', usecols=['title','abstract','authors','doi','publish_time'])

print (df.shape)

#drop duplicate abstracts

df = df.drop_duplicates(subset='title', keep="first")

#drop NANs 

df=df.dropna()

# convert abstracts to lowercase

df["abstract"] = df["abstract"].str.lower()



# search focus keeps abstracts with the anchor words covid,-cov-2,hcov2

df=search_focus(df)



#show 5 lines of the new dataframe

df.head()
###################### MAIN PROGRAM ###########################

questions = [

'Which patient populations have highest risk of death from COVID-19?',

'What is the average age of death cases?'

]



for question in questions:



    # remove punctuation, stop words and stem words from NL question

    search_words=remove_stopwords(question,stopwords)



     # get best sentences

    df_table=get_sentences(df,search_words)

    

    # sort df by sentence rank scores

    

    df_table=df_table.sort_values(by=['sent_score'], ascending=False)

    df_table=df_table.drop(['sent_score'], axis=1)

    

    #limit number of results

    df_answers=df_table.head(3)

    

    df_table=df_table.head(5)

    

    text=''

    

    for index, row in df_answers.iterrows():

        text=text+' '+row['excerpt']

    

    display(HTML('<h3>'+question+'</h3>'))

    

    display(HTML('<h4> Answer: '+answer_question(question,text)+'</h4>'))

    

    df_table=df_table.drop(['excerpt'], axis=1)

    

    #convert df to html

    df_table=HTML(df_table.to_html(escape=False,index=False))

    

    # show the HTML table with responses

    display(df_table)