!pip install summa


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import json

import glob

import sys

import urllib



import pickle

import gc

import json

import re 

import random



import os.path

from os import path

# load TextRank summarizer

from summa.summarizer import summarize

from summa.keywords import keywords
global parse_all

global url_link

global corpus_frac



root_path = '/kaggle/input/CORD-19-research-challenge'

corpus_frac = 1.0 #fraction of corpus to use



url_link={}

parse_all = False

# Just set up a quick blank dataframe to hold all these medical papers. 



corona_features = {"doc_id": [], "source": [], "title": [],

                  "abstract": [], "text_body": []}

corona_df = pd.DataFrame.from_dict(corona_features)



linkdict={}



json_filenames = glob.glob(f'{root_path}/**/*.json', recursive=True)

json_filenames = random.sample(json_filenames,int( len(json_filenames)*corpus_frac))
#support functions

def clean_dataset(text):

    if (not (isinstance(text, str)) ): return text

    text=re.sub("[\[].*?[\]]", "", text)#remove in-text citation

    text=re.sub(r'^https?:\/\/.*[\r\n]*', '',text, flags=re.MULTILINE)#remove hyperlink

    text=re.sub(r'^a1111111111 a1111111111 a1111111111 a1111111111 a1111111111.*[\r\n]*',' ',text)#have no idea what is a11111.. is, but I remove it now

    text=re.sub(' +', ' ',text ) #remove extra space, but was not able to remove all, see examples in the following cells

    text=re.sub(r's/ ( *)/\1/g','',text)

    

    return text

def return_append_metadata_df(df,linkdict):

    global url_link

    global parse_all

    global corpus_frac

    

    # load the meta data from the CSV file using 3 columns (abstract, title, authors),

    meta_df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv', 

                        usecols=['title','abstract','authors','doi','full_text_file','url'])

    #drop duplicates

    meta_df=meta_df.drop_duplicates()

    #drop NANs 

    meta_df=meta_df.dropna()

    # convert abstracts to lowercase

    #df["abstract"] = df["abstract"].str.lower()   

    lim=100000 * corpus_frac

    cnt=0

    for index, row in meta_df.iterrows():

        cnt+=1

        if (cnt>lim):break

        if ((cnt % 1000) ==0):

            print ("Load Metadata {}".format(cnt))

        

        new_row = {"doc_id": None, "source": None, "title": None,"authors": None,

              "abstract": None, "text_body": None, "paragraphs":[],"bibliography": None}

        new_row['paragraphs'].append( row['abstract'])

        new_row['title'] = row['title']

        new_row['authors']=row['authors']

        new_row['abstract']=row['abstract']

        new_row['text_body']=row['abstract']

        new_row['doc_id']=row['doi']

        new_row['source']=row['full_text_file']

        

        if (not(parse_all)):

                del new_row['source']

                del new_row['authors']

                del new_row['abstract']

                del new_row['text_body']

                del new_row['bibliography'] 

                

        linkdict[row['title'].lower()]=row['url'] #row['doi']

        url_link[row['title'].lower()]=row['url'] #'https://doi.org/'+row['doi']

        df = df.append(new_row,ignore_index=True)



    return df
import re



# Now we just iterate over the files and populate the data frame.

# using the JSON files

def return_corona_df(json_filenames, df, source,linkdict):

    global url_link

    global parse_all

    global corpus_frac

    lim=100000 * corpus_frac

    cnt=0



    for file_name in json_filenames:

        cnt+=1

        if (cnt>lim):break

        if ((cnt % 1000) ==0):

            print ("Load JSON {}".format(cnt))

        row = {"doc_id": None, "source": None, "title": None,"authors": None,

              "abstract": None, "text_body": None, "paragraphs":[],"bibliography": None}



        with open(file_name) as json_data:

            data = json.load(json_data)



            row['doc_id'] = data['paper_id']

            row['title'] = data['metadata']['title']

            

            lowTitle = row['title'].lower()

            linkdict[lowTitle]="0000"

            

            

            authors = ", ".join([author['first'] + " " + author['last'] \

                                 for author in data['metadata']['authors'] if data['metadata']['authors']])

            row['authors'] = authors

            bibliography = "\n ".join([bib['title'] + "," + bib['venue'] + "," + str(bib['year']) \

                                      for bib in data['bib_entries'].values()])

            row['bibliography'] = bibliography

            

            #find any DOI enties

            for bib in data['bib_entries'].values():

                bib_title_low=bib['title'].lower()

               # bib_data[lowTitle] = bib

                if ('other_ids' in bib):

                    ids = bib['other_ids']

                    if('DOI' in ids):

                        dois = ids['DOI']

                        for doi in dois:

                            linkdict[bib_title_low]=doi

                            #print ("{} -> {}".format(lowTitle,doi))

                    

            # Now need all of abstract. Put it all in 

            # a list then use str.join() to split it

            # into paragraphs. 

            abstract_list=[]

            if ('abstract' in data):

                abstract_list = [data['abstract'][x]['text'] for x in range(len(data['abstract']) - 1)]

            

            abstract = "\n ".join(abstract_list)



            row['abstract'] = abstract

            



            # And lastly the body of the text. For some reason I am getting an index error

            # In one of the Json files, so rather than have it wrapped in a lovely list

            # comprehension I've had to use a for loop like a neanderthal. 

            

            # Needless to say this bug will be revisited and conquered. 

            row['paragraphs']=abstract_list

            

            body_list = []

            for _ in range(len(data['body_text'])):

                try:

                    body_list.append(data['body_text'][_]['text'])

                    row['paragraphs'].append(data['body_text'][_]['text'])

                except:

                    pass



            body = "\n ".join(body_list)

            

            row['text_body'] = body

            



            

            # Augment the paragraphs with Textrank summaries

            extra_list=[]

            summary_threshold=2048

            #if (len(body)>summary_threshold):

            #    extra_list.append("TR1: " + summarize(body, ratio=0.1))

            #    extra_list.append("TR2: " + summarize(body, ratio=0.3))

            if (len(abstract)>summary_threshold):                

                extra_list.append("TR3: " + summarize(abstract, ratio=0.3))

            for subtext in row['paragraphs']:

                if (len(subtext)>summary_threshold):

                    extra_list.append("TR4: " + summarize(subtext, ratio=0.3))

            for subtext in extra_list:

                row['paragraphs'].append(subtext)

                

       

            #define links

            searchTitle = row['title']

            searchTitle = re.sub(r'\W+',' ', searchTitle)

            if (len(searchTitle)>160):

                p =searchTitle.find(' ',128)

                if (p>0):

                    searchTitle = searchTitle[0:p]

            qdict={'q': "!ducky filetype:pdf "+searchTitle}

            if (len(body_list)==0):

                #not body text -> assume no free pdf on web

                qdict={'q': "!ducky "+searchTitle}

            url_link[lowTitle]="https://duckduckgo.com/?"+urllib.parse.urlencode(qdict)





            # Now just add to the dataframe. 

            

            if source == 'b':

                row['source'] = "biorxiv_medrxiv"

            elif source == "c":

                row['source'] = "common_use_sub"

            elif source == "n":

                row['source'] = "non_common_use"

            elif source == "p":

                row['source'] = "pmc_custom_license"

                

            if (not(parse_all)):

                del row['source']

                del row['authors']

                del row['abstract']

                del row['text_body']

                del row['bibliography']



            df = df.append(row, ignore_index=True)

            

    return df

    



corona_df = return_corona_df(json_filenames, corona_df, 'b',linkdict)

corona_df = return_append_metadata_df( corona_df,linkdict)



# save

corona_out = corona_df.to_csv('kaggle_covid-19_open_csv_format.csv')

corona_pkl = corona_df.to_pickle('kaggle_covid-19_pickle.pkl')

# Store (serialize) the url_link dictionary

with open('url_links.pkl', 'wb') as handle:

    pickle.dump(url_link, handle, protocol=pickle.HIGHEST_PROTOCOL)
url_link

corona_df.head()

corona_df.tail()
