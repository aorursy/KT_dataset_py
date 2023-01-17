import numpy as np 

import pandas as pd

import dask.bag as db

import plotly.express as px
lines=db.read_text("/kaggle/input/arxiv/*.json") 

lines.take(2) ## Looks at first two records
import json



records=lines.map(lambda x:json.loads(x))



records.take(2)



print("Type of First Record After JSON LOADS ",type(lines.take(1)[0]))

print("Type of First Record After JSON LOADS ",type(records.take(1)[0]))
records_count=records.count()

records_count
print("Number of Records in ArXiv Data is ",records_count.compute())

records.map(lambda x:x['submitter']).frequencies(sort=True).compute() ### The map function here extracts only the submitter information and the frequencies function is applied on it
records.map(lambda x:x['submitter']).frequencies(sort=True).topk(k=10,key=1).compute()
records.map(lambda x:x['categories']).frequencies(sort=True).topk(k=20,key=1).compute()
extract_latest_version=lambda x:x['versions'][-1]["created"] ## Here -1 indicates the last element in the versions. 
data_after_2015=records.filter(lambda x:int(extract_latest_version(x).split(" ")[3])>=2015)
print("Number of Papers Published Since 2015 ",data_after_2015.count().compute())
data_after_2015.map(lambda x:x['categories']).frequencies(sort=True).topk(k=20,key=1).compute()
extract_latest_version_year=lambda x:x['versions'][-1]["created"].split(" ")[3]
pub_by_year=records.map(extract_latest_version_year).frequencies().to_dataframe(columns=['submission_year','num_submissions']).compute()
pub_by_year.head()
pub_by_year=pub_by_year.sort_values(by="submission_year")

pub_by_year.head()


px.line(x='submission_year',y='num_submissions',data_frame=pub_by_year,title="Distribution of Paper Published on Arxiv By Year")
ai_category_list=['stat.ML','cs.LG','cs.AI']
ai_docs = (records.filter(lambda x:any(ele in x['categories'] for ele in ai_category_list)==True))
print("Total Papers published in AI&ML ",ai_docs.count().compute())
ai_docs_by_year=ai_docs.map(extract_latest_version_year).frequencies().to_dataframe(columns=['submission_year','num_submissions']).compute()
ai_docs_by_year=ai_docs_by_year.sort_values(by="submission_year")

ai_docs_by_year.head()
px.line(x='submission_year',y='num_submissions',data_frame=ai_docs_by_year,title="AI & ML Paper Published on Arxiv By Year")
## Extracting author parsed for the first paper to look at the structure

authors=records.map(lambda x:x["authors_parsed"]).take(1)[0]

authors
[" ".join(a) for a in authors]
get_authors =lambda x: [' '.join(a).strip() for a in x['authors_parsed']]
ai_authors=ai_docs.map(get_authors).flatten().frequencies(sort=True).topk(k=20,key=1).to_dataframe(columns=['authors','num_submissions']).compute()
ai_authors.head()
ai_authors = ai_authors.sort_values('num_submissions', ascending=True)
px.bar(y="authors",x="num_submissions",data_frame=ai_authors,orientation="h")
get_metadata = lambda x: {'id': x['id'],

                  'title': x['title'],

                  'category':x['categories'],

                  'abstract':x['abstract'],

                 'version':x['versions'][-1]['created']}



ai_papers=ai_docs.map(get_metadata).to_dataframe().compute()
ai_papers.head()