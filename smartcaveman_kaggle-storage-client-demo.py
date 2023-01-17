![ ! -d client ] && git clone https://github.com/smartcaveman/kaggle-storage-client.git client
from kaggle_secrets import UserSecretsClient
from os import environ
user_secrets = UserSecretsClient()
environ['KAGGLE_USERNAME'] = user_secrets.get_secret('KAGGLE_USERNAME')
environ['KAGGLE_KEY']  = user_secrets.get_secret('KAGGLE_KEY')
from os import listdir
for folder in ["kaggle_storage_client","notebooks","tests"]:
    for file in filter(lambda f: ".py" in f or ".ipy" in f, listdir(f'client/{folder}')):
        with open(f'client/{folder}/{file}') as reader:
            content = reader.read() \
                            .replace('client.kaggle_storage_client', 'kaggle_storage_client') \
                            .replace('kaggle_storage_client', 'client.kaggle_storage_client') \
                            .replace('client.tests', 'tests') \
                            .replace('tests', 'client.tests')
        with open(f'client/{folder}/{file}', 'w') as writer:
            writer.write(content)
        print(f'Patched {folder}/{file}.')
import kaggle
import client.kaggle_storage_client
import client.tests
from client.kaggle_storage_client import LocalStorage, KaggleStorageClient, type_descriptor, DATASET_METADATA_FILE, DEFAULT_CONFIG_FILE, DEFAULT_DATA_DIR, print_fstree
from os import remove, path
from pandas import read_csv 
from client.tests import EXAMPLE_DATASET,  save_test_config_file
EXAMPLE_DATASET
# define local paths

local_file_1 = f'{DEFAULT_DATA_DIR}/{EXAMPLE_DATASET["OWNER"]}/{EXAMPLE_DATASET["NAME"]}/{EXAMPLE_DATASET["FILE_1"]}'
local_file_2 = f'{DEFAULT_DATA_DIR}/{EXAMPLE_DATASET["OWNER"]}/{EXAMPLE_DATASET["NAME"]}/{EXAMPLE_DATASET["FILE_2"]}' 
local_metadata_file = f'{DEFAULT_DATA_DIR}/{EXAMPLE_DATASET["OWNER"]}/{EXAMPLE_DATASET["NAME"]}/{DATASET_METADATA_FILE}'
# None of the local files exist at the start.  
#   - Local File #1 is downloaded from the remote dataset.
#   - Local File #2 is created locally then uploaded to the remote dataset.
for filepath in [local_file_1,local_file_2,local_metadata_file]:
    if path.exists(filepath) : os.remove(filepath)    
    assert not path.exists(filepath)
# create the configfile
configfile = save_test_config_file(configfile_content=f'{{username:{user_secrets.get_secret("KAGGLE_USERNAME")},key:{user_secrets.get_secret("KAGGLE_KEY")}}}')


# create the client instance
# - credentials are loaded from the local environment at '/root/.kaggle/kaggle.json' by default
# - datasets are downloaded to their corresponding paths in the relative 'data' folder by default

client = KaggleStorageClient(configfile=configfile)

t, m = type_descriptor(client)

# show the KaggleStorageClient API in a DataFrame view
data = [(t,mkey,(f'{msig}' if not f'{msig}'[0]=='(' else f'{msig} -> object') ) for (mkey,msig) in m]
pandas.DataFrame(data, columns=["Interface","Member","Signature"])
print_fstree(DEFAULT_DATA_DIR)
# generate local file #1
client.local_storage.save(  username=client.username, 
                            dataset=EXAMPLE_DATASET["NAME"], 
                            filename=EXAMPLE_DATASET["FILE_1"], 
                            content=EXAMPLE_DATASET["FILE_1_CONTENT"])
assert path.exists(local_file_1)

# upload the file from the local storage to the kaggle dataset
client.upload(EXAMPLE_DATASET["NAME"], local_file_1)

# show the generated file in a DataFrame
read_csv(local_file_1)
import time
time.sleep(1)
# downloads the remote example dataset file to the local file path
downloaded_file = client.download(
                                username=EXAMPLE_DATASET["OWNER"], 
                                dataset =EXAMPLE_DATASET["NAME"], 
                                filename=EXAMPLE_DATASET["FILE_1"])

# now the local file exists
assert path.exists(local_file_1)
# because
assert local_file_1 == downloaded_file.replace('\\','/')

# loads the example dataset from the downloaded file into a pandas DataFrame
read_csv(downloaded_file)
list(client.local_storage.files)
# generate local file #2
client.local_storage.save(  username=client.username, 
                            dataset=EXAMPLE_DATASET["NAME"], 
                            filename=EXAMPLE_DATASET["FILE_2"], 
                            content=EXAMPLE_DATASET["FILE_2_CONTENT"])
assert path.exists(local_file_2)

# upload the file from the local storage to the kaggle dataset
client.upload(EXAMPLE_DATASET["NAME"], local_file_2)

# show the generated file in a DataFrame
read_csv(local_file_2)