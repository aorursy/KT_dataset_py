# imports
import os
import glob
import json
import missingno as msno
import numpy as np
import pandas as pd
with open('/kaggle/input/CORD-19-research-challenge/metadata.readme', 'r') as f:
    print(f.read())
metadata_df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
metadata_df.head()
print(f'number of rows in metadata = {len(metadata_df):,}')
print(f'number of rows in metadata with (has_pdf_parse == True) = {sum(metadata_df["has_pdf_parse"] == True):,}')
print(f'number of rows in metadata with (has_pmc_xml_parse == True) = {sum(metadata_df["has_pmc_xml_parse"] == True):,}')
print(f'number of rows in metadata with (sha != NaN) = {sum(metadata_df["sha"].notna()):,}')
print(f'number of rows with unique sha = {len(metadata_df["sha"].unique()):,}')
duplicated_sha = metadata_df[(metadata_df['sha'].isna() == False) & metadata_df['sha'].duplicated()]['sha'].unique()
duplicated_sha
metadata_df[metadata_df['sha'].isin(duplicated_sha)].sort_values('sha')
metadata_df.drop(metadata_df[(metadata_df['sha'] == '45e40b2d7d973ed5c9798da613fb3cfa4427e2e2') & (metadata_df['title'] == 'Books received')].index, inplace=True)
print(f'number of rows in metadata after cleaning up duplicated sha = {len(metadata_df):,}')
with open('/kaggle/input/CORD-19-research-challenge/json_schema.txt', 'r') as f:
    print(f.read())
jsons = list()
for path in glob.iglob('/kaggle/input/CORD-19-research-challenge/**/*.json', recursive=True):
    sha = os.path.splitext(os.path.basename(path))[0]
    if sha in jsons:
        print(f'duplicate sha: {sha}')
    
    paper = {
        'sha': sha,
        'path': path,
    }
    
    with open(path, 'r') as f:
        json_data = json.load(f)
        paper['title'] = json_data['metadata']['title'] if json_data['metadata']['title'] else np.nan
        paper['authors'] = '; '.join([f'{author["first"]}{" " if author["first"] and author["middle"] else ""}{" ".join(author["middle"])}{" " if author["first"] or author["middle"] and author["last"] else ""}{author["last"]}' for author in json_data['metadata']['authors']]) if json_data['metadata']['authors'] else np.nan
        # NOTE: json_data does NOT match json_schema.
        # 'abstract', 'body_text', 'bib_entries', 'ref_entries' and 'back_matter' are NOT part of 'metadata'
        paper['abstract'] = '\n'.join([paragraph['text'] for paragraph in json_data['abstract']]) if 'abstract' in json_data and json_data['abstract'] else np.nan
        paper['body'] = '\n'.join([paragraph['text'] for paragraph in json_data['body_text']])
        
    jsons.append(paper)

jsons_df = pd.DataFrame(jsons)
jsons_df.replace()
jsons_df.head()
print(f'number of json files = {len(jsons_df):,}')
print(f'number of json files with unique sha = {len(jsons_df["sha"].unique()):,}')
metadata_cols = [
    'title',
    'authors',
    'abstract',
    'cord_uid',
    'source_x',
    'doi',
    'pmcid',
    'pubmed_id',
    'license',
    'publish_time',
    'journal',
    'Microsoft Academic Paper ID',
    'WHO #Covidence',
    'has_pdf_parse',
    'has_pmc_xml_parse',
    'url',
]
papers_df = jsons_df.set_index('sha').join(metadata_df.set_index('sha')[metadata_cols], how='left', rsuffix='_metadata')
papers_df.head()
print(f'number of rows in combined dataframe = {len(papers_df):,}')
print(f'number of extra rows (as a result of join, due to duplicate sha in metadata_df) in combined dataframe = {len(papers_df) - len(jsons_df):,}')
msno.matrix(papers_df)
print(f'number of NaN in title = {sum(papers_df["title"].isna()):,}')
print(f'number of NaN in authors = {sum(papers_df["authors"].isna()):,}')
print(f'number of NaN in abstract = {sum(papers_df["abstract"].isna()):,}')
print(f'number of NaN in title_metadata = {sum(papers_df["title_metadata"].isna()):,}')
print(f'number of NaN in authors_metadata = {sum(papers_df["authors_metadata"].isna()):,}')
print(f'number of NaN in abstract_metadata = {sum(papers_df["abstract_metadata"].isna()):,}')
papers_df['title'] = papers_df['title_metadata'].combine_first(papers_df['title'].replace('', np.nan))
papers_df['authors'] = papers_df['authors_metadata'].combine_first(papers_df['authors'].replace('', np.nan))
papers_df['abstract'] = papers_df['abstract_metadata'].combine_first(papers_df['abstract'].replace('', np.nan))
papers_df.drop(['title_metadata', 'authors_metadata', 'abstract_metadata'], axis=1, inplace=True)
papers_df.head()
msno.matrix(papers_df)
print(f'number of NaN in title = {sum(papers_df["title"].isna()):,}')
print(f'number of NaN in authors = {sum(papers_df["authors"].isna()):,}')
print(f'number of NaN in abstract = {sum(papers_df["abstract"].isna()):,}')
