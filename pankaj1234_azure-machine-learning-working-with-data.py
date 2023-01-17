# import the Azure ML libs.
!pip install azureml
!pip install azureml.core
!pip install azureml.widgets
!pip install azureml.train
!pip install azureml.dataprep

import azureml.core
import azureml.widgets 
print("Ready to use Azure ML", azureml.core.VERSION)
from azureml.core import Workspace
## In this segment you should replace the 3-parameters values according to the workspace available in the subscription
## ths experiment will not work beyond this point if these values are not appropriatly inserted.
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
# get the name of defult Datastore associated with the workspace.
default_dsname = ws.get_default_datastore().name
default_ds = ws.get_default_datastore()
print('default Datastore = ', default_dsname)
default_ds.upload_files(files=['../input/iris-flower-dataset/IRIS.csv'],
                 target_path='flower_data/',
                 overwrite=True, show_progress=True)

flower_data_ref = default_ds.path('flower_data').as_download('ex_flower_data')
print('reference path = ',flower_data_ref)
import os, shutil

# Create a folder for the experiment files
folder_name = 'datastore-experiment-files'
experiment_folder = './' + folder_name
os.makedirs(experiment_folder, exist_ok=True)
%%writefile $folder_name/iris_simple_experiment.py
from azureml.core import Run
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

#"argparse" to define the input parameters for the script.
import argparse

# Get the experiment run context -  we are going to pass this configuration later
run = Run.get_context()

#define the regularization parameter for the logistic regression.
parser = argparse.ArgumentParser()
parser.add_argument('--reg_rate', type=float, dest='reg', default=0.01)

#define the data_folder parameter for referencing the path of the registerd datafolder.
parser.add_argument('--data_folder', type=str, dest='data_folder', help='Data folder reference')
args=parser.parse_args()
r = args.reg
ex_data_folder = args.data_folder

#look into the files in datafolder
all_files = os.listdir(ex_data_folder)

# load the data from a local file
data = pd.concat((pd.read_csv(os.path.join(ex_data_folder,file)) for file in all_files))

X = data[['sepal_length', 'sepal_width','petal_length','petal_width']].values
X=StandardScaler().fit_transform(X)
Y= (data['species']).map(lambda x: 0 if x=='Iris-setosa' else (1 if x=='Iris-versicolor' else 2))

#Split data into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=1234)
# fit the model
run.log("model regularization", np.float(r))
model = LogisticRegression(C=1/r, solver='lbfgs', multi_class='multinomial').fit(X_train,Y_train)

Y_pred = model.predict(X_test)
accuracy = np.average(Y_test == Y_pred)
print("accuracy: " + str(accuracy))
run.log("Accuracy", np.float(accuracy))

# Save the trained model in the "outputs" folder. The "outputs" folder is standard output folder for AML.
os.makedirs("outputs", exist_ok=True)
joblib.dump(value=model, filename='outputs/iris_simple_model.pkl')
# Complete the run
run.complete()
from azureml.train.sklearn import SKLearn ## note - directly using SKLearn as estimator, hence avoid using conda_packages
from azureml.core import Experiment
from azureml.widgets import RunDetails

# Create an estimator, look into the 'data_folder' additional parameter and reference path as value
estimator = SKLearn(source_directory=experiment_folder,
                      entry_script='iris_simple_experiment.py',
                      compute_target='local',
                      use_docker=False,
                      script_params = {'--reg_rate': 0.07, '--data_folder':flower_data_ref} # assigned reference path value as defined above.
                      )

# Create an experiment
experiment_name = 'iris-datastore-experiment'
experiment = Experiment(workspace = ws, name = experiment_name)

# Run the experiment based on the estimator
run = experiment.submit(config=estimator)

# Get Run Details
RunDetails(run).show()

# Wait to complete the experiment. In the Azure Portal we will find the experiment state as preparing --> finished.
run.wait_for_completion(show_output=True)
from azureml.core import Dataset

# Creating tabular dataset from files in datastore.
tab_dataset = Dataset.Tabular.from_delimited_files(path=(default_ds,'flower_data/*.csv'))
tab_dataset.take(10).to_pandas_dataframe()

# similarly, creating files dataset from the files already in the datastore. Useful in scenarios like image processing in deeplearning.
file_dataset = Dataset.File.from_files(path=(default_ds,'flower_data/*.csv'))
for fp in file_dataset.to_path():
    print(fp)
# register tabular dataset
tab_dataset = tab_dataset.register(workspace=ws, name='flower tab ds', description='Iris flower Dataset in tabular format', tags={'format':'CSV'}, create_new_version=True)
#register File Dataset
file_dataset = file_dataset.register(workspace=ws, name='flower Files ds', description='Iris flower Dataset in Files format', tags={'format':'CSV'}, create_new_version=True)
print("Datasets:")
for dataset_name in list(ws.datasets.keys()):
    dataset = Dataset.get_by_name(ws, dataset_name)
    print("\t", dataset.name, 'version', dataset.version)
tab_ds_experiment = "tab_dataset_experiment"
os.makedirs(tab_ds_experiment,exist_ok=True)
print(tab_ds_experiment, 'created')
%%writefile $tab_ds_experiment/iris_simple_DTexperiment.py
from azureml.core import Run
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Get the experiment run context -  we are going to pass this configuration later
run = Run.get_context()

#define the regularization parameter for the logistic regression.
parser = argparse.ArgumentParser()
parser.add_argument('--reg_rate', type=float, dest='reg', default=0.01)
args=parser.parse_args()
r = args.reg

# load the data from a dataset -  passed as an "inputs" to the script
data = run.input_datasets['flower_ds'].to_pandas_dataframe()
X = data[['sepal_length', 'sepal_width','petal_length','petal_width']].values
X=StandardScaler().fit_transform(X)
Y= (data['species']).map(lambda x: 0 if x=='Iris-setosa' else (1 if x=='Iris-versicolor' else 2))

#Split data into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=1234)
# fit the model
model = LogisticRegression(C=1/r, solver='lbfgs', multi_class='multinomial').fit(X_train,Y_train)

Y_pred = model.predict(X_test)
accuracy = np.average(Y_test == Y_pred)
print("accuracy: " + str(accuracy))
run.log("Accuracy", np.float(accuracy))

# Save the trained model in the "outputs" folder. The "outputs" folder is standard output folder for AML.
os.makedirs("outputs", exist_ok=True)
joblib.dump(value=model, filename='outputs/iris_simple_DTmodel.pkl')
# Complete the run
run.complete()
from azureml.train.sklearn import SKLearn ## note - directly using SKLearn as estimator, hence avoid using conda_packages
from azureml.core import Experiment, Dataset
from azureml.widgets import RunDetails

# Get the previously registered tabular flower dataset
tab_dataset = ws.datasets.get('flower tab ds')

# Create an estimator, look into the 'data_folder' additional parameter and reference path as value
estimator = SKLearn(source_directory=tab_ds_experiment, ## pointing to the correct experiment folder for this context
                      entry_script='iris_simple_DTexperiment.py',
                      compute_target='local',
                      use_docker=False,
                      inputs=[tab_dataset.as_named_input('flower_ds')], ## pass the 'tab_dataset' as input to the experiment
                      pip_packages=['azureml-dataprep[pandas]']   ## passing azureml-dataprep to provision this package at runtime on execution env.
                      )

# Create an experiment
experiment_name = 'iris-dataset-experiment'
experiment = Experiment(workspace = ws, name = experiment_name)

# Run the experiment based on the estimator
run = experiment.submit(config=estimator)

# Get Run Details
RunDetails(run).show()

# Wait to complete the experiment. In the Azure Portal we will find the experiment state as preparing --> finished.
run.wait_for_completion(show_output=True)