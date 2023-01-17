# import the Azure ML libs.



!pip install azureml

!pip install azureml.core

!pip install azureml.widgets
import azureml.core

import azureml.widgets 

print("Ready to use Azure ML", azureml.core.VERSION)
from azureml.core import Workspace



## in this segment you should replace the 3-parameters values according to the workspace available in the subscription

## this experiment will not work beyond this point if these values are not appropriately inserted.

## HENCE, THE Notebook Execution will terminate





## Example - 

    ## ws = Workspace.get(name="<<MLSERVICENAME>>", subscription_id='<<GUID - ML Service ID>>', resource_group='<<Hosting Azure Resource Group>>')



# Pulling values from Kaggle Secrets



from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

MLServiceName = user_secrets.get_secret("MLServiceName")

az_resource_grp = user_secrets.get_secret("az_resource_grp")

sub_id = user_secrets.get_secret("sub_id")



## Instanciating the Workspace object.

ws = Workspace.get(name=MLServiceName, subscription_id=sub_id, resource_group=az_resource_grp)

print(ws.name, "loaded")
from azureml.core import Experiment

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline 



# Create an Azure ML experiment in your workspace

experiment = Experiment(workspace = ws, name = "simple-experiment")



# Start logging data from the experiment

run = experiment.start_logging()

print("Starting experiment:", experiment.name)



# load the data from a local file

data = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')



# Count the rows and log the result

row_count = (len(data))



# IMPORTANT: Log statistical value from this Experiment's run 

run.log('observations', row_count)

print('Analyzing {} rows of data'.format(row_count))



# IMPORTANT: Log plot image (log_image()) from this Experiment's run 

iris_counts = data['species'].value_counts()

fig = plt.figure(figsize=(6,6))

ax = fig.gca()    

iris_counts.plot.bar(ax = ax) 

ax.set_title('Count of Iris Species') 

ax.set_xlabel('Species') 

ax.set_ylabel('Instance Count')

plt.show()

run.log_image(name = 'label distribution', plot = fig)



# IMPORTANT: log distinct counts as a list using log_list from the Experiment's run

species = data['species'].unique()

run.log_list('species categories', species)



# IMPORTANT: log summary statistics as a row using log_row from the Experiment's run

features_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

summary_stats = data[features_columns].describe().to_dict()

for col in summary_stats:

    keys = list(summary_stats[col].keys())

    values = list(summary_stats[col].values())

    for index in range(len(keys)):

        run.log_row(col, stat = keys[index], value = values[index])

        

# Save a sample of the data and upload it to the experiment output

data.sample(100).to_csv('../sample.csv', index=False, header=True)

run.upload_file(name = 'outputs/sample.csv', path_or_stream = '../sample.csv')



# Complete the run

run.complete()
import json



# Get run details

details = run.get_details()

print(details)



# Get logged metrics

metrics = run.get_metrics()

print(json.dumps(metrics, indent=2))



# Get output files

files = run.get_file_names()

print(json.dumps(files, indent=2))