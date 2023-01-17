# Installs
#!pip list
#!pip install pandas
#!pip install numpy
#!pip install pycountry
#!pip install plotly
#!pip install matplotlib
import pandas as pd
import numpy as np
import pycountry as pc
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import warnings
warnings.filterwarnings('ignore')
import json
#with open("/tf/CORD-19_20200410/Corona2.json") as f:
#https://www.kaggle.com/medical-ner/Corona2.json
with open("/kaggle/input/medical-ner/Corona2.json") as f:
    annotation = json.load(f)
TRAIN_DATA  = []
for e in annotation["examples"]:
    content = e["content"]
    entities = []
    for an in e["annotations"]:        
        if len(an["value"]) == len(an["value"].strip()):          
            if len(an['human_annotations']) == 0:
                continue
            info = (an["start"],an["end"],an["tag_name"])
            entities.append(info)
            #print(an["start"],an["end"],an["tag_name"])
    if len(entities) > 0:
        TRAIN_DATA.append(([content,{"entities":entities}]))    
from __future__ import unicode_literals, print_function
import random
from pathlib import Path
from spacy.util import minibatch, compounding
import spacy
import sys
spacy.util.use_gpu(0)
#def train_model(model=None, output_dir="/tf/CORD-19_20200410/medical-ner", n_iter=1000):
def train_model(model=None, output_dir="/kaggle/working/medical-ner", n_iter=1000):
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        if model is None:
            nlp.begin_training(device=0)
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 64.0, 1.2))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  
                    annotations,  
                    drop=0.20, 
                    losses=losses
                   
                )
            print("Losses", losses)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)
#train_model()
#nlp2 = spacy.load("/tf/CORD-19_20200410/medical-ner")
#nlp2 = spacy.load("/kaggle/working/medical-ner")
nlp2 = spacy.load("/kaggle/input/cord19-temporal-geotagger-dataset/medical-ner")
import numpy as np
import pandas as pd
import os
import csv
import json
import random
#metadata = open('/tf/CORD-19_20200410/metadata.csv', 'r')
metadata = open('/kaggle/input/CORD-19-research-challenge/metadata.csv', 'r')
metadata_fieldnames=('cord_uid','sha','source_x','title','doi','pmcid','pubmed_id','license','abstract','publish_time','authors','journal','Microsoft Academic Paper ID','WHO #Covidence','has_pdf_parse','has_pmc_xml_parse','full_text_file','url')
metadata_reader = csv.DictReader(metadata, metadata_fieldnames)
cord_uid = []
sha = []
title = []
pmcid = []
pubmed_id = []
abstract = []
publish_time = []
for row in metadata_reader:
    cord_uid.append(row['cord_uid'].strip())
    sha.append(row['sha'].strip())
    title.append(row['title'].strip())
    pmcid.append(row['pmcid'].strip())
    pubmed_id.append(row['pubmed_id'].strip())
    abstract.append(row['abstract'].strip())
    publish_time.append(row['publish_time'].strip())
metadata.close()
    
files = []
#for dirname, _, filenames in os.walk('/tf/CORD-19_20200410/dat/'):
for dirname, _, filenames in os.walk('/kaggle/input/CORD-19-research-challenge/'):
    for filename in filenames:
        if ".json" in filename:           
            fpath = os.path.join(dirname, filename)
            if len(files) < 300:
                files.append(fpath)
random.shuffle(files)
# Local transmission of SARS took place in Toronto, Ottawa, San Francisco, Ulaanbaatar, Manila, Singapore, Taiwan, Hanoi and Hong Kong
# within China it spread to Guangdong, Jilin, Hebei, Hubei, Shaanxi, Jiangsu, Shanxi, Tianjin, and Inner Mongolia.
regions_of_interest = [['China','China'],
                       ['Hong Kong','China'],
                       ['Mongolia','Mongolia'],
                       ['Ulaanbaatar','Mongolia'],
                       ['Philippines','Philippines'],
                       ['Manila','Philippines'],
                       ['Taiwan','Taiwan'],
                       ['Vietnam','Vietnam'],
                       ['Hanoi','Vietnam'],
                       ['Canada','Canada'],
                       ['Toronto','Canada'],
                       ['Ottawa','Canada'],
                       ['Singapore','Thailand'],
                       ['Taiwan','Taiwan'],
                       ['San Francisco','US']]
country_date = [['','','']]
first_country_date = True
output = []
entities = []
for i in range(0,len(files)):
    if i%100 == 0:
        print('completed ', i)
    with open(files[i]) as f:
        file_data = json.load(f)
    paper_id = file_data["paper_id"]
    for o in file_data["body_text"]: 
            doc = nlp2(o["text"],disable=['parser','tagger'])
            for ent in doc.ents:
                #if len(ent.text) > 2:
                if ent.text == 'SARS-CoV':
                    entities.append((ent.text, ent.label_))
                    #print(paper_id)
                    index_paper_id = -1
                    if paper_id in pmcid:
                        index_paper_id = pmcid.index(paper_id)
                        #print('index_pmcid=' + str(index_paper_id))
                    elif paper_id in pubmed_id:
                        index_paper_id = pubmed_id.index(paper_id)
                        #print('index_pubmed_id=' + str(index_paper_id))
                    elif paper_id in sha:
                        index_paper_id = sha.index(paper_id)
                        #print('index_sha=' + str(index_paper_id))
                    if index_paper_id >= 0:
                        for roi in regions_of_interest:
                            region = roi[0]
                            country = roi[1]
                            if (region in title[index_paper_id]) or (region in abstract[index_paper_id]) or (region in ent.text):
                                #cases_time_ner.write("%s,%s,1,0,,,0,\n" % (country,publish_time[index_paper_id]))
                                if first_country_date:
                                    country_date[0][0] = country
                                    country_date[0][1] = publish_time[index_paper_id]
                                    country_date[0][2] = cord_uid[index_paper_id]
                                    first_country_date = False
                                else:
                                    country_date.append([country,publish_time[index_paper_id],cord_uid[index_paper_id]])
cases_time_ner_fieldnames='Country_Region,Last_Update,Publications,CORD_UID,Recovered,Active,Delta_Confirmed,Delta_Recovered'
#cases_time_ner = open('/tf/CORD-19_20200410/cases_time_ner.csv', 'w')
cases_time_ner = open('/kaggle/working/cases_time_ner_300.csv', 'w')
cases_time_ner.write("%s\n" % cases_time_ner_fieldnames)
from operator import itemgetter
dates = []
countries = []
country_count = []
date_country_count = [['','',0,'']]
cd_sorted_by_country = sorted(country_date, key=itemgetter(0))
cd_sorted_by_date = sorted(country_date, key=itemgetter(1))
first_date_country_count = True
for sorted_by_country in cd_sorted_by_country:
    country = sorted_by_country[0]
    if country not in countries:
        countries.append(country)
        country_count.append(0)

counter = 0
previous_date = ''
for sorted_by_date in cd_sorted_by_date:
    date = sorted_by_date[1]
    date_country = sorted_by_date[0]
    date_cord_uid = sorted_by_date[2]
    if date not in dates:
        dates.append(date)
    for country in countries:
        if previous_date != date:
            if first_date_country_count:
                date_country_count[0][0] = date
                date_country_count[0][1] = country
                date_country_count[0][2] = country_count[countries.index(country)]
                date_country_count[0][3] = date_cord_uid
                first_date_country_count = False
            else:
                date_country_count.append([date,country,country_count[countries.index(country)],date_cord_uid])
            if country == date_country:
                country_count[countries.index(country)] += 1
                date_country_count[counter][2] = country_count[countries.index(country)]
            counter += 1
        elif country == date_country:
            country_count[countries.index(country)] += 1
            counter_index = counter - len(countries) + countries.index(country)
            date_country_count[counter_index][2] = country_count[countries.index(country)]
    previous_date = date
            
counter = 0
for date in dates:
    for country in countries:
        publications = date_country_count[counter][2]
        cord_uid = date_country_count[counter][3]
        cases_time_ner.write("%s,%s,%d,%s,,,0,\n" % (country,date,publications,cord_uid))
        counter += 1
cases_time_ner.close()
#df_table = pd.read_csv("/tf/CORD-19_20200410/cases_time_ner.csv",parse_dates=['Last_Update'])
df_table = pd.read_csv("/kaggle/input/cord19-temporal-geotagger-dataset/cases_time_ner.csv",parse_dates=['Last_Update'])
df_data = df_table.groupby(['Last_Update', 'Country_Region'])['Publications', 'CORD_UID'].max().reset_index()
df_data["Last_Update"] = pd.to_datetime( df_data["Last_Update"]).dt.strftime('%m/%d/%Y')

fig = px.scatter_geo(df_data,
                     locations              = "Country_Region",
                     locationmode           = 'country names', 
                     color                  = np.power(df_data["Publications"],0.3)-2,
                     size                   = np.power(df_data["Publications"]+1,0.3)-1,
                     hover_name             = "Country_Region",
                     hover_data             = ["Publications","CORD_UID"],
                     range_color            = [0, max(np.power(df_data["Publications"],0.3))], 
                     projection             = "natural earth",
                     animation_frame        = "Last_Update", 
                     color_continuous_scale = px.colors.sequential.Plasma,
                     title                  = 'SARS-CoV: Progression of Spread Publication Coverage'
                    )
fig.update_coloraxes(colorscale="hot")
fig.update(layout_coloraxis_showscale=False)
fig.show()
for n in range(15):
    slide = '/kaggle/input/cord19-temporal-geotagger-dataset/CORD-19_Temporal_Geotagger_and_Analysis_of_Risk_Factors_20-0800/Slide' + str(n+1) + '.JPG'
    plt.figure(figsize=(12,9))
    plt.axis('off')
    img=mpimg.imread(slide)
    imgplot = plt.imshow(img)
    plt.show()
