"""

Added 3 columns to the metadata.csv file (4/03/2020): 





    PT : List of publication types extracted from Medline record for PMID.  

    review : 0/1 flag for articles where PT list includes "Review"

    systematic_review_or_ma : 0/1 flag for articles where 1 or more of the following:

        -PT list includes "Systematic Review" and/or "Meta-Analysis"

        -Article title includes "Systematic Review" and/or "Meta-Analysis" 

"""



import pandas as pd

import json

from Bio import Entrez

from Bio import Medline

from collections import Counter



Entrez.email = 'mayars169@gmail.com'
#Read metadata into a pandas dataframe and extract non-null pubmed IDs.



metadata_df = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv')



index_pmids_df = metadata_df[metadata_df.pubmed_id.notnull()]['pubmed_id'].astype('int').astype('str')



index_pmids_dict = index_pmids_df.to_dict()

#Overview of metadata pubmed IDs.



print("Metadata dataframe has %d rows." % len(metadata_df))

print("Metadata dataframe has %d rows with non-null pubmed IDs." %len(index_pmids_df))

print("Metadata dataframe has %d rows with NaN pubmed IDs." % len(metadata_df.loc[metadata_df.pubmed_id.isna()]))

print("Metadata dataframe has %d duplicate rows." % (len(metadata_df) - len(metadata_df.drop_duplicates())))
#What do the duplicate rows look like?

#3/27 version of metadata no longer has duplicate rows





(metadata_df[metadata_df.duplicated()]).head()
"""



Using Entrez, download medline records for metadata_df non-null pubmed IDs.

ISSN codes for the journal names will be pulled from these records.

This took ~5 minutes on my home internet.



Was unable to run this through Kaggle, so uploaded a pregenerated copy of pmid_issn_dict as json.



"""

"""

idlist = index_pmids_df.tolist()



records = []

for retstart in [0, 10000, 20000]:

    print("%d records downloaded..." % retstart)

    handle = Entrez.efetch(db="pubmed",id=idlist, rettype="medline", retstart=retstart, retmode="json")

    records += list(Medline.parse(handle))



"""

"""    

Generate a dict mapping pubmed IDs to ISSN codes.



Some records do not include a PMID and/or IS field.

"""

"""

pmid_issn_dict = {}



#Categorize entries based on contents of PT field in record



pmid_pt_dict = {}

for record in records:

    if 'PMID' in record:  

        record_pt = record.get('PT', [])

        if record_pt:

            

            review_flag = 0

            sr_flag = 0

            ma_flag = 0

            

            if 'Review' in record_pt:

                review_flag = 1

            if 'Systematic Review' in record_pt:

                sr_flag = 1

            if 'Meta-Analysis' in record_pt:

                ma_flag = 1

            

            pmid_pt_dict[record['PMID']] = {

                'PT' : record['PT'],

                'review' : review_flag,

                'systematic_review_or_ma' : 1 if (sr_flag + ma_flag) > 0 else 0   

            }

   

with open('resources/pmid_pt_dict.json', 'w') as f:

    json.dump(pmid_pt_dict, f)



"""
"""

article_types = set([])



for record in records:

    for record_article_type in record.get('PT', []):

        article_types.add(record_article_type)



print("Article types:")

for article_type in article_types:

    print(article_type)



Article types:

Historical Article

Evaluation Study

Guideline

Lecture

Journal Article

Retracted Publication

Editorial

Congress

Introductory Journal Article

Systematic Review

Meta-Analysis

Research Support, U.S. Gov't, P.H.S.

Clinical Conference

Clinical Trial, Phase I

Video-Audio Media

Comparative Study

Research Support, N.I.H., Extramural

Case Reports

Book Chapter

Randomized Controlled Trial, Veterinary

English Abstract

Autobiography

Directory

Research Support, N.I.H., Intramural

Clinical Trial, Phase II

Clinical Trial, Phase IV

Observational Study

Clinical Trial, Phase III

Research Support, Non-U.S. Gov't

Published Erratum

Pragmatic Clinical Trial

Comment

Observational Study, Veterinary

Clinical Trial Protocol

Dataset

Research Support, U.S. Gov't, Non-P.H.S.

Research Support, American Recovery and Reinvestment Act

Patient Education Handout

Overall

Address

Validation Study

Biography

Portrait

Letter

Randomized Controlled Trial

Practice Guideline

Multicenter Study

News

Controlled Clinical Trial

Review

Clinical Trial, Veterinary

Retraction of Publication

Consensus Development Conference

Clinical Trial

Interview

Bibliography

Corrected and Republished Article



"""
with open('../input/pmid-pt-dict/pmid_pt_dict.json') as f:

    pmid_pt_dict = json.load(f)



pmid_pub_type_df = pd.DataFrame.from_dict(pmid_pt_dict).transpose()



pmid_pub_type_df['pubmed_id'] = pmid_pub_type_df.index.astype(int)

metadata_pt_type_df = pd.merge(metadata_df, pmid_pub_type_df, on='pubmed_id', how='left')
#Breakdown of systematic_review_or_ma flag

metadata_pt_type_df.systematic_review_or_ma.value_counts(dropna=False)
"""

Replace title null values with '' and lowercase titles for searching.

"""



metadata_pt_type_df['edited_title'] = metadata_pt_type_df.title.fillna('')

metadata_pt_type_df['edited_title'] = metadata_pt_type_df['edited_title'].str.lower()
#Systematic review or meta-analysis flag for rows where title contains 'systematic review'

metadata_pt_type_df[metadata_pt_type_df.edited_title.str.contains('systematic review')].systematic_review_or_ma.value_counts()
#Systematic review or meta-analysis flag for rows where title contains 'meta-analysis'

metadata_pt_type_df[metadata_pt_type_df.edited_title.str.contains('meta-analysis')].systematic_review_or_ma.value_counts()
metadata_pt_type_df.loc[metadata_pt_type_df.edited_title.str.contains('systematic review'), 'systematic_review_or_ma'] = 1

metadata_pt_type_df.loc[metadata_pt_type_df.edited_title.str.contains('meta-analysis'), 'systematic_review_or_ma'] = 1
#Drop edited title

metadata_pt_type_df = metadata_pt_type_df.drop(['edited_title'], axis=1)
#Post title-searching breakdown of systemic_review_or_ma flag

metadata_pt_type_df.systematic_review_or_ma.value_counts(dropna=False)
print("Number of rows in metadata_pt_type_df: %d" % len(metadata_pt_type_df))

print("Number of rows with null PMID: %d" % len(metadata_pt_type_df.loc[metadata_pt_type_df.pubmed_id.isna()]))

print("Number of rows with non-null PMID: %d" % len(metadata_pt_type_df.loc[~metadata_pt_type_df.pubmed_id.isna()]))



print("Number of rows with systemic_review_or_ma = 1: %d" % len(metadata_pt_type_df.loc[metadata_pt_type_df.systematic_review_or_ma == 1]))

print("Number of rows with systemic_review_or_ma = 0: %d" % len(metadata_pt_type_df.loc[metadata_pt_type_df.systematic_review_or_ma == 0]))

print("Number of rows with systemic_review_or_ma = null: %d" % len(metadata_pt_type_df.loc[metadata_pt_type_df.systematic_review_or_ma.isna()]))



print("Number of systemic_review_or_ma rows with non-null PMID: %d" % len(metadata_pt_type_df.loc[metadata_pt_type_df.systematic_review_or_ma == 1].loc[~metadata_pt_type_df.pubmed_id.isna()]))

print("Number of systemic_review_or_ma rows with null PMID: %d" % len(metadata_pt_type_df.loc[metadata_pt_type_df.systematic_review_or_ma == 1].loc[metadata_pt_type_df.pubmed_id.isna()]))
metadata_pt_type_df.to_csv('metadata_pub_type_df_190403.csv', index=False)
filtered_metadata_pt_type_df = metadata_pt_type_df.loc[metadata_pt_type_df.systematic_review_or_ma == 1]
filtered_metadata_pt_type_df.to_csv('metadata_pub_type_df_filtered_190403.csv', index=False)