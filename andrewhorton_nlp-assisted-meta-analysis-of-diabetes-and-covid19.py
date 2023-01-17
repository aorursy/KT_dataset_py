import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandasql import sqldf # sql access to pandas dataframes

sql = lambda q: sqldf(q, globals())

from collections import defaultdict

import json #utilized when parsing the json files associated with the challenge articles

from smart_open import open

import os



from wordcloud import WordCloud, STOPWORDS #For creating/visualizing Word Clouds in the notebook



import spacy #Entity Extraction/Labeling

from spacy.pipeline import EntityRuler,Sentencizer

from spacy.lang.en import English

from spacy import displacy #Utilized to explore entity identification in-line with the documents themselves



import matplotlib #Meta-Analysis Visualizations

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

%matplotlib inline



#Adjust the parameters below, when required, to help review dataframes easier.  

##With a fail amount of the data being long text fields, for example, adjusting the column widths will help when trying to manually review data.

pd.set_option('display.max_colwidth',100)

pd.set_option("display.max_rows", 20)
metadata_df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv', low_memory=False)
#Extracting the year from the publish_date field, and then summarizing to better understand the provided dataset

PublishDates=pd.DataFrame(pd.DatetimeIndex(metadata_df['publish_time']).year)

PublishDates_df2=sql("""SELECT * from (select publish_time as year, count(publish_time) as article_count from PublishDates group by publish_time) where article_count >500 """)
#Creating a simple bar chart to review how many articles are present by year for which they were published

figure(figsize=(15,10))



y_pos = np.arange(len(PublishDates_df2['year']))

frequency = PublishDates_df2['article_count']





plt.bar(y_pos, frequency, align='center', alpha=0.5)

plt.xticks(y_pos, PublishDates_df2['year'])

plt.ylabel('Article Count')

plt.xlabel('Year')

plt.title('Article Count by Published Year')



plt.show()
article_selection_df1 = sql("""

SELECT row_number() OVER () - 1 as id, *

FROM metadata_df 

WHERE abstract is not NULL

AND (has_pdf_parse = True OR has_pmc_xml_parse = True)

AND (pubmed_id >= 31775236

     OR (pubmed_id is null AND (publish_time like '%2020%' or date(publish_time) >='2019-12-01')))

""")

##PubMed IDs were utilized as a means to identify relevant articles since the IDs are sequential AND not all dates in the metadata file are accurate
article_selection_df2 = article_selection_df1.assign(sha=article_selection_df1.sha.str.split(';'))

article_selection_df2 = article_selection_df2.explode('sha')

#Difference from the first metadata file to now (changes made 4/3).  Two different paths now for different json files

article_selection_df2_pdf = sql("""

SELECT '../input/CORD-19-research-challenge/' || full_text_file || '/' || full_text_file || '/' || 'pdf_json/' || trim(sha) || '.json' as filename

, *

FROM article_selection_df2

WHERE has_pdf_parse = True

""")

article_selection_df2_pmc = sql("""

SELECT '../input/CORD-19-research-challenge/' || full_text_file || '/' || full_text_file || '/' || 'pmc_json/' || trim(pmcid) || '.xml.json' as filename

, *

FROM article_selection_df2

WHERE has_pmc_xml_parse = True

""")

article_selection_df3=article_selection_df2_pdf.append(article_selection_df2_pmc)
article_selection_df3 = article_selection_df3.assign(file_exists=lambda x: x.filename.apply(os.path.exists))

article_selection_df3 = sql("""

SELECT filename, file_exists, cord_uid, sha, source_x, title, doi, pmcid, pubmed_id, license, abstract, publish_time, authors, journal, full_text_file, url

FROM article_selection_df3

WHERE file_exists = True

""")
filenames = article_selection_df3.filename.values

file_data = ( json.load(open(f, mode='r')) for f in filenames )

docs = {'abstracts': list(), 'full_text': list()}

for jdoc in file_data:

    docs['full_text'].append( ' '.join([ para['text'] for para in jdoc['body_text'] ]) )

    docs['abstracts'].append( ' '.join([ para['text'] for para in jdoc.get('abstract', []) ]) )

article_selection_df4 = article_selection_df3.assign(full_text=docs['full_text'], json_abstract=docs['abstracts'])
#Word Cloud for the full text extracted from the json files

json_full_text = article_selection_df4.full_text.values

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(str(json_full_text))

fig = plt.figure(

    figsize = (15, 10),

    facecolor = 'k',

    edgecolor = 'k')

plt.subplot(1,2,1)

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)



#Word Cloud for the abstract text from the metadata file

abstract_text = article_selection_df4.abstract.values

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(str(abstract_text))

fig = plt.figure(

    figsize = (15, 10),

    facecolor = 'k',

    edgecolor = 'k')

plt.subplot(1,2,2)

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)



#A word cloud for the extracted abstract text was not included as it was similar to the results from the metadata file.



#showing both of the word clouds defined above

plt.show()
article_selection_df5 = sql("""

SELECT filename, file_exists, cord_uid, sha, source_x, title, doi, pmcid, pubmed_id, license, abstract as meta_abs,length(abstract) as meta_abs_len,json_abstract, 

        length(json_abstract) as json_abs_len,publish_time, authors, journal, full_text_file, url, full_text

FROM article_selection_df4 

WHERE ((abstract like '%covid%' or abstract like '%novel coronavirus%'or abstract like '%SARS-COV-2%'or abstract like '%SARS-COV2%'or abstract like '%SARSCOV2%'

    or abstract like '%2019-nCoV%' or abstract like '%2019nCoV%')

    and

    (abstract like '%diabetes%' or abstract like '%diabetic%'or abstract like '%hyperglycemia%'or abstract like '%blood sugar%'or abstract like '%HbA1c%'

    or abstract like '%Hemoglobin A1c%' or abstract like '%heart failure%'or abstract like '%hypertension%'or abstract like '%hypertensive%'or abstract like '%blood pressure%')

    and 

    (abstract like '%death%' or abstract like '%fatality%' or abstract like '%non-survivor%' or abstract like '%nonsurvivor%' or abstract like '%fatal%' or abstract like '%fatalities%'))

    OR

    ((json_abstract like '%covid%' or json_abstract like '%novel coronavirus%'or json_abstract like '%SARS-COV-2%'or json_abstract like '%SARS-COV2%'or json_abstract like '%SARSCOV2%'

    or json_abstract like '%2019-nCoV%' or json_abstract like '%2019nCoV%')

    and

    (json_abstract like '%diabetes%' or json_abstract like '%diabetic%'or json_abstract like '%hyperglycemia%'or json_abstract like '%blood sugar%'or json_abstract like '%HbA1c%'

    or json_abstract like '%Hemoglobin A1c%' or json_abstract like '%heart failure%'or json_abstract like '%hypertension%'or json_abstract like '%hypertensive%'or json_abstract like '%blood pressure%')

    and 

    (json_abstract like '%death%' or json_abstract like '%fatality%' or json_abstract like '%non-survivor%' or json_abstract like '%nonsurvivor%' or json_abstract like '%fatal%' or json_abstract like '%fatalities%'))

    OR 

    ((full_text like '%covid%' or full_text like '%novel coronavirus%'or full_text like '%SARS-COV-2%'or full_text like '%SARS-COV2%'or full_text like '%SARSCOV2%'

    or full_text like '%2019-nCoV%' or full_text like '%2019nCoV%')

    and

    (full_text like '%diabetes%' or full_text like '%diabetic%'or full_text like '%hyperglycemia%'or full_text like '%blood sugar%'or full_text like '%HbA1c%'

    or full_text like '%Hemoglobin A1c%' or full_text like '%heart failure%'or full_text like '%hypertension%'or full_text like '%hypertensive%'or full_text like '%blood pressure%')

    and 

    (full_text like '%death%' or full_text like '%fatality%' or full_text like '%non-survivor%' or full_text like '%nonsurvivor%' or full_text like '%fatal%' or full_text like '%fatalities%'))

""")
ArticleSelectionSummary=sql("""

SELECT InitialCount,DateFilterCount,WithJsonFilterCount,WithFullTextCount,FinalListCount from

(SELECT count(*) as InitialCount from metadata_df) a join 

(SELECT count(*) as DateFilterCount from article_selection_df1) b join

(SELECT count(*) as WithJsonFilterCount from article_selection_df2) c join

(SELECT count(*) as WithFullTextCount from article_selection_df4) d join

(SELECT count(*) as FinalListCount from article_selection_df5) e

""")

#WithJsonFilterCount shows an increase due to splitting those articles listed with multiple sha's; to ensure all relevant data is abstracted in following processes

##WithFullTextCount shows an increase due to the articles having both sets of json data



ArticleSelectionSummary
nlp = English()

ruler = EntityRuler(nlp)

nlp.add_pipe(nlp.create_pipe('sentencizer'))

patterns = [{"label": "nFatal", "pattern": [{'LIKE_NUM': True,'OP':'+'}, {'LOWER': 'non','OP':'+'}, {'IS_PUNCT': True,'OP':'+'}, {'LOWER': 'survivors','OP':'+'}],'id':'n_fatalities'},

            {"label": "nFatal", "pattern": [{'LIKE_NUM': True,'OP':'+'}, {'LOWER': 'fatalities','OP':'+'}],'id':'n_fatalities'},

            {"label": "nFatal", "pattern": [{'LIKE_NUM': True,'OP':'+'}, {'LOWER': 'cases','OP':'+'},{'LOWER': 'with','OP':'+'},{'LOWER': 'fatalities','OP':'+'}],'id':'n_fatalities'},

            {"label": "nFatal", "pattern": [{'LIKE_NUM': True,'OP':'+'}, {'LOWER': 'cases','OP':'+'},{'LOWER': 'died','OP':'+'}],'id':'n_fatalities'},

            {"label": "nFatal", "pattern": [{'LIKE_NUM': True,'OP':'+'}, {'LOWER': 'deaths','OP':'+'}],'id':'n_fatalities'},

            {"label": "nFatal", "pattern": [{'LIKE_NUM': True,'OP':'+'}, {'LOWER': 'died','OP':'+'}],'id':'n_fatalities'},

            {"label": "nFatal", "pattern": [{'LIKE_NUM': True,'OP':'+'}, {'LOWER': 'fatal','OP':'+'}],'id':'n_fatalities'},

            {"label": "nFatal", "pattern": [{'LIKE_NUM': True,'OP':'+'}, {'LOWER': 'patients','OP':'+'},{'LOWER': 'who','OP':'+'},{'LOWER': 'died','OP':'+'}],'id':'n_fatalities'},                                         

     

            {"label": "n", "pattern": [{'LIKE_NUM': True,'OP':'+'}, {'LOWER': 'patients','OP':'+'}],'id':'total patients'},

            {"label": "n", "pattern": [{'LIKE_NUM': True,'OP':'+'}, {'LOWER': 'and','OP':'+'},{'LIKE_NUM': True,'OP':'+'}, {'LOWER': 'patients','OP':'+'}],'id':'total patients'},

            {"label": "n", "pattern": [{'LIKE_NUM': True,'OP':'+'}, {'LOWER': 'cases','OP':'+'}],'id':'total patients'},

            

            {"label": "nper", "pattern": [{'LIKE_NUM': True,'OP':'+'},{'ORTH': '%','OP':'+'},{'LOWER': 'diabetes','OP':'+'},{'ORTH': ',','OP':'+'}],'id':'comorbidity percent'},

            {"label": "nper", "pattern": [{'LOWER': 'diabetes','OP':'+'},{'ORTH': '(','OP':'+'},{'LIKE_NUM': True,'OP':'+'},{'IS_PUNCT': True,'OP':'+'},{'LIKE_NUM': True,'OP':'+'},{'IS_PUNCT': True,'OP':'+'}],'id':'comorbidity percent'},

            {"label": "nper", "pattern": [{'LOWER': 'diabetes','OP':'+'},{'ORTH': '(','OP':'+'},{'LIKE_NUM': True,'OP':'+'},{'ORTH': '%','OP':'+'}],'id':'comorbidity percent'},

           ]

ruler.add_patterns(patterns)

nlp.add_pipe(ruler)
displacy_example_df=article_selection_df5[article_selection_df5['sha'] =='ad3bfbf5daf646a4ba1be6495c8b83c797944577']

data = []

docs_in = nlp.pipe(displacy_example_df['full_text'].values)

docs_out = []

titles_in = displacy_example_df['title'].values

for i, doc in enumerate(docs_in):

    if len(doc.ents) > 1:

        doc.user_data['title'] = titles_in[i]

        docs_out.append(doc)

        for ent in doc.ents:

            data.append({'id': i,'entity_value': ent.text, 'entity_type': ent.label_})



displacy.render(docs_out, style='ent', page=True, jupyter=True)
entity_extraction=[]

docs_in = nlp.pipe(article_selection_df5['full_text'].values)

docs_out = []

sha_in = article_selection_df5['sha'].values

titles_in = article_selection_df5['title'].values

for i, doc in enumerate(docs_in):

    if len(doc.ents) > 1:

        doc.user_data['sha'] = sha_in[i]

        doc.user_data['title'] = titles_in[i]

        docs_out.append(doc)

        for j, sent in enumerate(doc.sents):

            sent_data = {}

            for ent in sent.ents:

                if sent_data.get(ent.label_) is None:

                    sent_data[ent.label_] = ent.text

                else:

                    sent_data[ent.label_] = sent_data[ent.label_] + ';' + ent.text

            entity_extraction.append({'docID': i,'sha':sha_in[i],'title':titles_in[i],'entity_sentencizer':sent.text,'sentID': j, **sent_data})

entity_extraction_df1=pd.DataFrame(entity_extraction)
same_sentence_extract=sql("""select sha,title,sentID, 'SameSentence' as StatsExtraction,nFatal, nper from entity_extraction_df1 

                            where nFatal is not null AND nper is not null""")



diff_sentence_extract=sql("""select a.sha,a.title,a.sentID,'DiffSentence' as StatsExtraction, a.nFatal,b.nper from 

           entity_extraction_df1 a join entity_extraction_df1 b 

           on a.sha=b.sha and a.nFatal is not null and b.nper is not null and (b.sentID>a.sentID AND b.sentID - a.sentID < 10)

          """)

#prefer the same sentence findings, if possible, so removing them from the second dataframe (the multi-sentence results)

diff_sentence_extract=diff_sentence_extract[~diff_sentence_extract['sha'].isin(same_sentence_extract['sha'])]



entity_extraction_df2=same_sentence_extract.append(diff_sentence_extract)
nper = entity_extraction_df2['nper']

comorbidities = ['diabetes']

metrics = []

como_types = []

for ent in nper:

    for mo in comorbidities:

        if mo in ent:

            como_types.append(mo)

            metrics.append(ent.replace(mo, '').strip())



data_intermediate = {'Comorbidity_Type': como_types, 'metrics': metrics}

df_int = pd.DataFrame(data_intermediate)

entity_cleansing_df1=pd.merge(entity_extraction_df2,df_int,on=nper)

entity_cleansing_df1=entity_cleansing_df1.drop_duplicates()
entity_cleansing_df1=entity_cleansing_df1[entity_cleansing_df1.sha != '875b7c463f00772fa0dc18ada678bc1ff16a4274']
entity_cleansing_df1
entity_cleansing_df1['Fatalities Number']=entity_cleansing_df1['nFatal'].str.split(" ",expand=True)[0]

#pulling out the diabetes percentages (Some statistics have both the number and %, some are just the %, etc.)

entity_cleansing_df1['Comorbidity Percentage']=entity_cleansing_df1['metrics'].replace(to_replace=r'\([1-9][0-9]/[1-9][0-9]',value='',regex=True).replace(to_replace=r'\(',value='',regex=True).replace(to_replace=r'\)',value='',regex=True).replace(to_replace=r'[1-9][0-9]-',value='',regex=True).replace(to_replace=r'[0-9],',value='',regex=True).replace(to_replace=r',',value='',regex=True).replace(to_replace=r'%',value='',regex=True)

entity_cleansing_df1['Comorbidity Percentage']=entity_cleansing_df1['Comorbidity Percentage'].astype(float).div(100).round(4)



entities_extracted=entity_cleansing_df1[['sha','title','Comorbidity_Type','Fatalities Number','Comorbidity Percentage']].drop_duplicates()



#Converting the Fatalities Number column to int such that the following meta-analysis steps could be executed; previously the dtype was defined as an object

entities_extracted=entities_extracted.astype({'Fatalities Number': 'int32'})
entities_extracted
#Create Confidence intervals and standard errors

meta_analysis_df=pd.DataFrame(entities_extracted)

meta_analysis_df['L_95CL'] = meta_analysis_df['Comorbidity Percentage']-(1.96*np.sqrt((meta_analysis_df['Comorbidity Percentage']*(1-meta_analysis_df['Comorbidity Percentage']))/meta_analysis_df['Fatalities Number']))

meta_analysis_df['U_95CL'] = meta_analysis_df['Comorbidity Percentage']+(1.96*np.sqrt((meta_analysis_df['Comorbidity Percentage']*(1-meta_analysis_df['Comorbidity Percentage']))/meta_analysis_df['Fatalities Number']))

meta_analysis_df['SE'] = np.sqrt((meta_analysis_df['Comorbidity Percentage']*(1-meta_analysis_df['Comorbidity Percentage']))/meta_analysis_df['Fatalities Number'])

meta_analysis_df['Study_ID'] = np.arange(len(meta_analysis_df))
#Compute pooled estimate and 95% CI

total_deaths=sum(meta_analysis_df['Fatalities Number'])

tot_deaths=str(total_deaths)



total_deaths_diab=sum(meta_analysis_df['Fatalities Number']*meta_analysis_df['Comorbidity Percentage'])

tot_deaths_diab=str(round(total_deaths_diab))



prev_diab_deaths= round((total_deaths_diab/total_deaths)*100,2)

pr_diab_deaths=str(prev_diab_deaths)

prev_diab_deaths_num=prev_diab_deaths/100



LCL95= round(prev_diab_deaths_num-(1.96*np.sqrt((prev_diab_deaths_num*(1-prev_diab_deaths_num))/total_deaths)),4)

LCL95_char=str(LCL95*100)

UCL95= round(prev_diab_deaths_num+(1.96*np.sqrt((prev_diab_deaths_num*(1-prev_diab_deaths_num))/total_deaths)),4)

UCL95_char=str(UCL95*100)





print("Total Deaths: " + tot_deaths)

print("Total Deaths with Diabetes: " + tot_deaths_diab)

print("Pooled Prevalence of Diabetes in Fatal Cases: " + pr_diab_deaths + "% , 95% CI: ( " + LCL95_char + "% ," + UCL95_char + "% )" )
# Build the plot

fig, ax = plt.subplots()

ax.bar(meta_analysis_df['Study_ID'], meta_analysis_df['Comorbidity Percentage'], yerr=meta_analysis_df['SE'], align='center', alpha=0.5, ecolor='black', capsize=10)

ax.set_ylabel('Proportion of Deaths with Diabetes')

ax.set_xticks(meta_analysis_df['Study_ID'])

ax.set_xticklabels(meta_analysis_df['Study_ID'])

ax.set_xlabel('Study')

ax.set_title('Diabetes Prevalence in Fatal COVID-19 Cases by Study')

ax.yaxis.grid(True)



#Add Horizontal Line for pooled estimate and Confidence Limits

ax.axhline(LCL95, color='green', linewidth=1)

ax.axhline(prev_diab_deaths_num, color='darkgreen', linewidth=3)

ax.axhline(UCL95, color='green', linewidth=1)
