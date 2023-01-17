!pip install pyspark



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(color_codes=True)



import glob

import json



from pathlib import Path



root_path = Path('/kaggle/input/CORD-19-research-challenge/2020-03-13')

metadata_path = root_path / Path('all_sources_metadata_2020-03-13.csv')



metadata = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})



metadata.rename(columns={'source_x': 'source', 'Microsoft Academic Paper ID': 'mic_id', 'WHO #Covidence': 'who_covidence'}, inplace=True)



print("There are ", len(metadata), " sources in the metadata file.")



metadata.head(2)
all_json = glob.glob( str(root_path) + '/**/*.json', recursive=True)

print("There are ", len(all_json), "sources files.")
from pyspark import SparkContext

from pyspark.sql import SparkSession

from pyspark.sql.functions import *



spark = (

    SparkSession.builder.appName("covid")

    .master("local[*]")

    .config("spark.driver.memory", "16g")

    .config("spark.executor.memory", "16g")

    .config("spark.driver.maxResultSize", "4g")

    .getOrCreate()

)



data = spark.read.json(all_json, multiLine=True)

data.createOrReplaceTempView("data")



#data.printSchema()
# Select text columns

covid_sql = spark.sql(

        """

        SELECT

            metadata.title AS title,

            abstract.text AS abstract, 

            body_text.text AS body_text,

            back_matter.text AS back_matter,

            paper_id

        FROM data

        """)



# Convert it to pandas and join all texts

covid_pd = covid_sql.toPandas()

covid_pd['abstract'] = covid_pd['abstract'].str.join(' ')

covid_pd['body_text'] = covid_pd['body_text'].str.join(' ')

covid_pd['back_matter'] = covid_pd['back_matter'].str.join(' ')



covid_pd.head()

covid_pd.to_csv('clean_covid.csv', index=False)