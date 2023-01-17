"""

Added 3 columns to the metadata.csv file (3/24/2020): ISSN, SJR, and H index from the 2018 SCIMAGO table.



ISSN - Journal identifier code



SJR (SCImago Journal Rank) indicator

    It expresses the average number of weighted citations received in the selected year by the documents 

    published in the selected journal in the three previous years, --i.e. weighted citations received in 

    year X to documents published in the journal in years X-1, X-2 and X-3. See detailed description of SJR (PDF)

    .

H Index

    The h index expresses the journal's number of articles (h) that have received at least h citations. 

    It quantifies both journal scientific productivity and scientific impact and it is also applicable to 

    scientists, countries, etc. (see H-index wikipedia definition)



In brief, the steps were:



    -Query Entrez for 22943 PMIDs extracted from metadata.csv

    -Extract ISSN codes from downloaded Medline records

    -Map PMIDs -> ISSN codes -> SJR/H index from downloaded SCIMAGO 2018 table.

    -Map journal names (for rows with journal but no PMID) -> SJR/H index.

    -Add and populate the 3 additional columns to metadata.csv



I was able to populate SJR and H index for 26358/26359 rows.



Of the 17862 unavailable, 11047 were due to the metadata.csv row lacking both a pubmed_id and journal name.

    -It may be possible to resolve some of these by querying Entrez for article name or other details.



The remainder are mostly due to SCIMAGO not listing the journal in 2018 (I am not clear why this is).

    -It may be possible to resolve some of these by downloading earlier versions of the SCIMAGO table and using the latest available listing.

    

"""



import pandas as pd

import json

from Bio import Entrez

from Bio import Medline

from collections import Counter



#Entrez.email = '#########'
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



for record in records:

    if 'PMID' in record:

        if 'IS' in record:

            if record['PMID'] in pmid_issn_dict:

                print(record['PMID'])

            pmid_issn_dict[record['PMID']] = record['IS']

        else:

            print(record)

    else:

        print(record)

        

with open('resources/pmid_issn_dict.json', 'w') as f:

    json.dump(pmid_issn_dict, f)



print("Number of PMIDs submitted to Entrez: %d" % len(idlist))

print("Number of records downloaded: %d" % len(records))

print("Number of records containing PMID and ISSN: %d" % len(pmid_issn_dict))



#Number of PMIDs submitted to Entrez: 22943

#Number of records downloaded: 22943

#Number of records containing PMID and ISSN: 22921



"""
"""

Example entry: '0002-8703 (Print) 0002-8703 (Linking)'



All cases have 1 ID followed by '(Print)' or '(Electronic)' and most have a 2nd ID followed by '(Linking)'



From testing, it appears that SJR uses the Linking ID where available.



"""



with open('../input/covid-if-resources/pmid_issn_dict.json') as f:

    pmid_issn_dict = json.load(f)
pmid_issn_fm_dict = {}

for pmid, issn in pmid_issn_dict.items():

    if 'Linking' in issn:

        pmid_issn_fm_dict[pmid] = issn.split(' ')[2].replace('-','')

    else:

        pmid_issn_fm_dict[pmid] = issn.split(' ')[0].replace('-','')
"""

Scientific Journal rankings table downloaded from portal at:

https://www.scimagojr.com/



"The SCImago Institutions Rankings (SIR) is a classification of academic and research-related 

institutions ranked by a composite indicator that combines three different sets of indicators 

based on research performance, innovation outputs and societal impact measured by their web visibility."



The two main ranking indicators are:



SJR (SCImago Journal Rank) indicator

    It expresses the average number of weighted citations received in the selected year by the 

    documents published in the selected journal in the three previous years, --i.e. weighted citations 

    received in year X to documents published in the journal in years X-1, X-2 and X-3. See detailed 

    description of SJR (PDF).

H Index

    The h index expresses the journal's number of articles (h) that have received at least h citations. 

    It quantifies both journal scientific productivity and scientific impact and it is also applicable to 

    scientists, countries, etc. (see H-index wikipedia definition)



https://www.scimagojr.com/help.php



"""



scimagojr_df = pd.read_csv('../input/scimagojr2018/scimagojr 2018.csv', delimiter=';')
scimagojr_df.head()
#The 'Issn' field contains 1 8-digit ISSN code, or 2 separated by ", "



print(scimagojr_df['Issn'].head())
#Splitting the code into 2 columns, and creating a new dataframe with columns of interest.



issn_split_df = scimagojr_df["Issn"].str.split(", ", n = 1, expand = True) 



scimagojr_df['issn_a'] = issn_split_df[0]

scimagojr_df['issn_b'] = issn_split_df[1]



scimagojr_trunc_df = scimagojr_df[['Issn', 'issn_a', 'issn_b', 'Title', 'SJR', 'H index']]
print(scimagojr_trunc_df.loc[scimagojr_trunc_df.issn_a == '00257753'])
#Converting the dataframe into a dict keyed to issn_a and issn_b



scimagojr_trunc_issn_a_df = scimagojr_trunc_df.set_index('issn_a').transpose()

scimagojr_trunc_issn_a_dict = scimagojr_trunc_issn_a_df.to_dict()



scimagojr_trunc_issn_b_df = scimagojr_trunc_df.set_index('issn_b').transpose()

scimagojr_trunc_issn_b_dict = scimagojr_trunc_issn_b_df.to_dict()



scimagojr_trunc_issn_merge_dict = {**scimagojr_trunc_issn_a_dict , **scimagojr_trunc_issn_b_dict}
#Map pmids to scimago fields



pmid_stats_dict = {}

not_found_issns = []

for pmid, issn in pmid_issn_fm_dict.items():

    if (scimagojr_trunc_issn_merge_dict.get(issn)):

        pmid_stats_dict[pmid] = scimagojr_trunc_issn_merge_dict[issn]

    else:

        print("Issn %s not found for pmid %s" % (issn, pmid))

        not_found_issns.append(issn)
"""

Issns not found in Scimago (2018) table.



I checked a few of the most common omissions and they can be found on the Scimago portal and in 

previous versions of the table, but not the 2018 one.  It's not immediately apparent to me why: at least one of 

the journals is still being published.



Example misssing ISSNs:



00653527 - Advances in Virus Research, on scimago website, 1981-ongoing

00063002 - Biochimica et biophysica acta, not on scimago website

20734409 - Cells, on scimago website, 2015-ongoing

20551169 - Journal of feline medicine and surgery open reports, not on scimago website

11107243 - Journal of Biomedicine and Biotechnology, not on scimago website



"""



missing_issn_pmid_counter = Counter(not_found_issns)

missing_issn_pmid_counts = [i[1] for i in missing_issn_pmid_counter.most_common()]



print("Number of unique ISSN codes not found: %d" % len(missing_issn_pmid_counter))

print("Number of PMIDs with ISSN codes not found: %d" % sum(missing_issn_pmid_counts))

#Map abbreviated journal names to SJR fields



pmid_journal_df = metadata_df[['pubmed_id', 'journal']]

pmid_journal_df = pmid_journal_df.dropna()

pmid_journal_dict = dict(zip(pmid_journal_df.pubmed_id, pmid_journal_df.journal))



journal_stats_dict = {}



for pmid, stats in pmid_stats_dict.items():

    

    journal = pmid_journal_dict.get(float(pmid))

    if journal:

        journal_stats_dict[journal] = stats



journal_stats_df = pd.DataFrame.from_dict(journal_stats_dict).transpose()

journal_stats_df['journal']= journal_stats_df.index



joined_df = pd.merge(metadata_df, journal_stats_df, on='journal', how='left')
print("Length of metadata dataframe: %d" % len(metadata_df))

print("Length of joined dataframe: %d" % len(joined_df))



print("Length of de-duplicated metadata dataframe: %d" % len(metadata_df.drop_duplicates()))

print("Length of de-duplicated joined dataframe: %d" % len(joined_df.drop_duplicates()))
#Filtering and reordering columns



metadata_if_df_cols =  metadata_df.columns.to_list() + ['Issn', 'SJR', 'H index']



metadata_if_df = joined_df.drop(['issn_b', 'issn_a'], axis=1)

metadata_if_df = metadata_if_df[metadata_if_df_cols]
"""

Missing impact factor breakdown

"""



print("Length of metadata_if_df: %d" % len(metadata_if_df))

print("Rows in metadata_if_df with SJR score: %d" % len(metadata_if_df.loc[~metadata_if_df.SJR.isna()]))

print("Rows in metadata_if_df with no SJR score: %d" % len(metadata_if_df.loc[metadata_if_df.SJR.isna()]))

print("Rows in metadata_if_df with H index score: %d" % len(metadata_if_df.loc[~metadata_if_df['H index'].isna()]))

print("Rows in metadata_if_df with no H index score: %d" % len(metadata_if_df.loc[metadata_if_df['H index'].isna()]))

print("Rows in metadata_if_df with Pubmed ID: %d" % len(metadata_if_df.loc[~metadata_if_df.pubmed_id.isna()]))

print("Rows in metadata_if_df with no Pubmed ID: %d" % len(metadata_if_df.loc[metadata_if_df.pubmed_id.isna()]))

print("Rows in metadata_if_df with journal name: %d" % len(metadata_if_df.loc[~metadata_if_df.journal.isna()]))

print("Rows in metadata_if_df with no journal name: %d" % len(metadata_if_df.loc[metadata_if_df.journal.isna()]))

print("Rows in metadata_if_df with no Pubmed ID or journal name: %d" % len(metadata_df.loc[metadata_df.pubmed_id.isna()].loc[metadata_df.journal.isna()]))

print("Rows in metadata_if_df with Pubmed ID and no SJR score: %d" % len(metadata_if_df.loc[metadata_if_df.SJR.isna()].loc[~metadata_if_df.pubmed_id.isna()]))

print("Rows in metadata_if_df with Pubmed ID and no H index score: %d" % len(metadata_if_df.loc[metadata_if_df['H index'].isna()].loc[~metadata_if_df.pubmed_id.isna()]))
metadata_if_df.to_csv('metadata_sjr_if_df_200324.csv', index=False)