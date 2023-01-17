#Featuretools is a framework to perform automated feature engineering.
#It excels at transforming temporal and relational datasets into feature matrices for machine learning.
#https://docs.featuretools.com/index.html
import featuretools as ft
#Load Mock Data
data = ft.demo.load_mock_customer()
#Prepare data
customers_df = data["customers"]
customers_df
sessions_df = data["sessions"]
sessions_df.sample(5)
transactions_df = data["transactions"]
transactions_df.sample(5)
#First, we specify a dictionary with all the entities in our dataset.
entities = {
    "customers" : (customers_df, "customer_id"),
    "sessions" : (sessions_df, "session_id", "session_start"),
    "transactions" : (transactions_df, "transaction_id", "transaction_time")
    }
#In this dataset we have two relationships
relationships = [("sessions", "session_id", "transactions", "session_id"),
                 ("customers", "customer_id", "sessions", "customer_id")]
#Run Deep Feature Synthesis
feature_matrix_customers, features_defs = ft.dfs(entities=entities,
                                                 relationships=relationships,
                                                 target_entity="customers")
feature_matrix_customers
#Change target entity
feature_matrix_sessions, features_defs = ft.dfs(entities=entities,
                                                relationships=relationships,
                                                target_entity="sessions")
feature_matrix_sessions.head(5)

#I just copy the code of Intro about featuretool. I will update for this kennel soon