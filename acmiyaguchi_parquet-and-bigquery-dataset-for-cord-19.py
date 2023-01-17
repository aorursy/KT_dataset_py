# These resources must be created by hand, either by the gcloud cli or via the GCP console

PROJECT_ID = "cord-19-271603"

BUCKET_NAME = "acmiyaguchi-cord-19-data"

DATASET_ID = "kaggle"
! pip install pyspark
from pyspark.sql.functions import lit

from pyspark.sql.types import (

    ArrayType,

    IntegerType,

    MapType,

    StringType,

    StructField,

    StructType,

)





def generate_cord19_schema():

    """Generate a Spark schema based on the semi-textual description of CORD-19 Dataset.



    This captures most of the structure from the crawled documents, and has been

    tested with the 2020-03-13 dump provided by the CORD-19 Kaggle competition.

    The schema is available at [1], and is also provided in a copy of the

    challenge dataset.



    One improvement that could be made to the original schema is to write it as

    JSON schema, which could be used to validate the structure of the dumps. I

    also noticed that the schema incorrectly nests fields that appear after the

    `metadata` section e.g. `abstract`.

    

    [1] https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-13/json_schema.txt

    """



    # shared by `metadata.authors` and `bib_entries.[].authors`

    author_fields = [

        StructField("first", StringType()),

        StructField("middle", ArrayType(StringType())),

        StructField("last", StringType()),

        StructField("suffix", StringType()),

    ]



    authors_schema = ArrayType(

        StructType(

            author_fields

            + [

                StructField(

                    "affiliation",

                    StructType(

                        [

                            StructField("laboratory", StringType()),

                            StructField("institution", StringType()),

                            StructField(

                                "location",

                                StructType(

                                    [

                                        StructField("settlement", StringType()),

                                        StructField("country", StringType()),

                                    ]

                                ),

                            ),

                        ]

                    ),

                ),

                StructField("email", StringType()),

            ]

        )

    )



    # used in `section_schema` for citations, references, and equations

    spans_schema = ArrayType(

        StructType(

            [

                # character indices of inline citations

                StructField("start", IntegerType()),

                StructField("end", IntegerType()),

                StructField("text", StringType()),

                StructField("ref_id", StringType()),

            ]

        )

    )



    # A section of the paper, which includes the abstract, body, and back matter.

    section_schema = ArrayType(

        StructType(

            [

                StructField("text", StringType()),

                StructField("cite_spans", spans_schema),

                StructField("ref_spans", spans_schema),

                # Equations don't appear in the abstract, but appear here

                # for consistency

                StructField("eq_spans", spans_schema),

                StructField("section", StringType()),

            ]

        )

    )



    bib_schema = MapType(

        StringType(),

        StructType(

            [

                StructField("ref_id", StringType()),

                StructField("title", StringType()),

                StructField("authors", ArrayType(StructType(author_fields))),

                StructField("year", IntegerType()),

                StructField("venue", StringType()),

                StructField("volume", StringType()),

                StructField("issn", StringType()),

                StructField("pages", StringType()),

                StructField(

                    "other_ids",

                    StructType([StructField("DOI", ArrayType(StringType()))]),

                ),

            ]

        ),

        True,

    )



    # Can be one of table or figure captions

    ref_schema = MapType(

        StringType(),

        StructType(

            [

                StructField("text", StringType()),

                # Likely equation spans, not included in source schema, but

                # appears in JSON

                StructField("latex", StringType()),

                StructField("type", StringType()),

            ]

        ),

    )



    return StructType(

        [

            StructField("paper_id", StringType()),

            StructField(

                "metadata",

                StructType(

                    [

                        StructField("title", StringType()),

                        StructField("authors", authors_schema),

                    ]

                ),

                True,

            ),

            StructField("abstract", section_schema),

            StructField("body_text", section_schema),

            StructField("bib_entries", bib_schema),

            StructField("ref_entries", ref_schema),

            StructField("back_matter", section_schema),

        ]

    )





def extract_dataframe_kaggle(spark):

    """Extract a structured DataFrame from the semi-structured document dump.



    It should be fairly straightforward to modify this once there are new

    documents available. The date of availability (`crawl_date`) and `source`

    are available as metadata.

    """

    base = "/kaggle/input/CORD-19-research-challenge"

    crawled_date = "2020-03-20"

    sources = [

        "noncomm_use_subset",

        "comm_use_subset",

        "biorxiv_medrxiv",

        "custom_license",

    ]



    dataframe = None

    for source in sources:

        path = f"{base}/{source}/{source}"

        df = (

            spark.read.json(path, schema=generate_cord19_schema(), multiLine=True)

            .withColumn("crawled_date", lit(crawled_date))

            .withColumn("source", lit(source))

        )

        if not dataframe:

            dataframe = df

        else:

            dataframe = dataframe.union(df)

    return dataframe

! ls /kaggle/input/CORD-19-research-challenge
from pyspark.sql import SparkSession



spark = SparkSession.builder.getOrCreate()

df = extract_dataframe_kaggle(spark)



output_path = "/kaggle/working/cord19"

%time df.coalesce(1).write.parquet(output_path, mode="overwrite")



! ls -alh /kaggle/working/cord19

! cp /kaggle/working/cord19/*.parquet ./cord19.parquet
import pandas as pd

import re



def snake_case(name):

    return "_".join([word.lower() for word in re.split(r"\W", name) if word])



input_path = "/kaggle/input/CORD-19-research-challenge/metadata.csv"

output_path = "metadata.parquet"



df = pd.read_csv(input_path)

df.columns = [snake_case(name) for name in df.columns]

%time df.to_parquet(output_path)
! ls -alh .
# https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python

# https://cloud.google.com/bigquery/docs/loading-data-cloud-storage-parquet

import glob

from google.cloud import storage

from google.cloud import bigquery



def upload(

    project,

    bucket_name,

    source_file_name,

    destination_blob_name,

    dataset_id,

    table_id,

    source_format=bigquery.SourceFormat.PARQUET

):

    storage_client = storage.Client(project=project)

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    uri = f"gs://{bucket_name}/{destination_blob_name}"

    print(f"copied {source_file_name} to {uri}")



    bigquery_client = bigquery.Client(project=PROJECT_ID)

    dataset_ref = bigquery_client.dataset(dataset_id)



    job_config = bigquery.LoadJobConfig()

    job_config.source_format = source_format

    job_config.write_disposition = bigquery.job.WriteDisposition.WRITE_TRUNCATE



    print(f"loading {uri} into {project}:{dataset_id}.{table_id}")

    load_job = bigquery_client.load_table_from_uri(

        uri, dataset_ref.table(table_id), job_config=job_config

    )

    load_job.result()
upload(

    PROJECT_ID, 

    BUCKET_NAME, 

    "cord19.parquet", 

    "2020-03-20_cord19.parquet", 

    DATASET_ID,

    "cord19",

)
upload(

    PROJECT_ID, 

    BUCKET_NAME, 

    "metadata.parquet", 

    "2020-03-20_metadata.parquet", 

    DATASET_ID,

    "metadata",

)
client = bigquery.Client(project=PROJECT_ID)
query = """

WITH

  cited_publications AS (

  SELECT

    metadata.title AS dst_title,

    bib_entry.value.title AS src_title

  FROM

    `cord-19-271603.kaggle.cord19`,

    UNNEST(bib_entries.key_value) AS bib_entry )

SELECT

  src_title AS title,

  COUNT(DISTINCT dst_title) AS num_referenced

FROM

  cited_publications

GROUP BY

  title

ORDER BY

  num_referenced DESC

LIMIT 20

"""



# Query complete (4.8 sec elapsed, 63.3 MB processed)

for row in client.query(query).result():

    print(f"{row.num_referenced}: {row.title[:80]}")
query = """

SELECT

  source_x,

  SUM(CAST(has_full_text AS INT64)) AS num_full_text,

  COUNT(*) AS total_documents

FROM

  `cord-19-271603.kaggle.metadata`

GROUP BY

  1

"""



# Query complete (0.4 sec elapsed, 164.7 KB processed)

for row in client.query(query).result():

    print(f"{row.source_x}: {row.num_full_text}/{row.total_documents} have text")