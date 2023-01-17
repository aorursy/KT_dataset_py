# Identify the project ID containing the desired dataset for analysis in this kernel

PROJECT_ID = 'my-example-dataset'



# Import the BQ API Client library

from google.cloud import bigquery

client = bigquery.Client(project=PROJECT_ID, location='US')



# Construct a reference to the Ames Housing dataset that is within the project

dataset_ref = client.dataset('ameshousing', project=PROJECT_ID)



# Make an API request to fetch the dataset

dataset = client.get_dataset(dataset_ref)
# Make a list of all the tables in the dataset

tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset

for table in tables:  

    print(table.table_id)
# Preview the first five lines of the table

client.list_rows(table, max_results=5).to_dataframe()
# What are the most and least common residential home styles in this dataset? 

# For each home style, how many do and do not have central Air Conditioning?



# Write the query

query1 = """

          SELECT 

            DISTINCT HouseStyle AS HousingStyle, 

            CentralAir AS HasAirConditioning,

            COUNT(HouseStyle) AS Count

          FROM 

            `my-example-dataset.ameshousing.train` 

          GROUP BY 

            HousingStyle, 

            HasAirConditioning

          ORDER BY 

            HousingStyle, 

            HasAirConditioning DESC

        """
# Set up the query

query_job1 = client.query(query1)



# Make an API request  to run the query and return a pandas DataFrame

housestyleAC = query_job1.to_dataframe()



# See the resulting table made from the query

print(housestyleAC)
# Create a linear model that trains on the variables GrLivArea, YearBuilt, OverallCond, OverallQual.

# GrLivArea = Above grade (ground) living area square feet

# YearBuilt = Year the home was completed

# OverallCond = Overall condition of the home

# OverallQual = Overall quality of the home



model1 = """

          CREATE OR REPLACE MODEL 

            `my-example-dataset.ameshousing.linearmodel`

          OPTIONS(model_type='linear_reg', ls_init_learn_rate=.15, l1_reg=1, max_iterations=5) AS

          SELECT 

            IFNULL(SalePrice, 0) AS label,

            IFNULL(GrLivArea, 0) AS LivingAreaSize,

            YearBuilt, 

            OverallCond, 

            OverallQual

          FROM 

            `my-example-dataset.ameshousing.train`

          """
# Set up the query

query_job2 = client.query(model1)



# Make an API request  to run the query and return a pandas DataFrame

linearmodel_GrLivArea = query_job2.to_dataframe()



# See the resulting table made from the query

# print(linearmodel_GrLivArea)
 ## Step 5: Create a Model using BigQuery ML

model1_stats = """

          SELECT

            *

          FROM 

            ML.TRAINING_INFO(MODEL `my-example-dataset.ameshousing.linearmodel`)

        """    
# Set up the query

query_job3 = client.query(model1_stats)



# Make an API request  to run the query and return a pandas DataFrame

linearmodel_stats = query_job3.to_dataframe()



# See the resulting table made from the query

print(linearmodel_stats)