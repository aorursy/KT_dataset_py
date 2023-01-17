import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import git
import os

directory = "abhi"
parent_dir = "/kaggle/working"

try:
    BASE_PATH = os.path.join(parent_dir, directory)  
    os.mkdir(BASE_PATH) 
    print("Directory '%s' created" %directory) 
except:
    pass

git.Git(BASE_PATH).clone("https://github.com/abhisha1991/w251_hw6")
print("Repository is created")
# steps taken from here: https://cloud.ibm.com/docs/cloud-object-storage?topic=cloud-object-storage-python

!pip install ibm-cos-sdk
import boto3
from botocore.client import Config
import ibm_boto3
from ibm_botocore.client import Config, ClientError

# Constants for IBM COS values - BAD PRACTICE TO EXPOSE THE VALUES
COS_ENDPOINT = "https://s3.eu-de.cloud-object-storage.appdomain.cloud"
COS_API_KEY_ID = "<<REDACTED>>"
COS_SERVICE_CRN = "crn:v1:bluemix:public:iam-identity::a/f3de9bdab4ee4727b5f81c87efa6d530::serviceid:ServiceId-2ef85b26-fc98-4444-bc26-2af926723bfa"
COS_AUTH_ENDPOINT = "https://iam.bluemix.net/oidc/token"
COS_RESOURCE_CRN = "crn:v1:bluemix:public:cloud-object-storage:global:a/f3de9bdab4ee4727b5f81c87efa6d530:1f545981-7f91-47a0-8437-6b4c581e31e2::"
BUCKET_NAME = "abhihw6bucket"

# Create resource
cos = ibm_boto3.resource("s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_service_instance_id=COS_RESOURCE_CRN,
    ibm_auth_endpoint=COS_AUTH_ENDPOINT,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)

'''
# Create client 
cos = ibm_boto3.client("s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_service_instance_id=COS_SERVICE_CRN,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)
'''

def get_bucket_item_list(bucket_name):
    print("Retrieving bucket contents from: {0}".format(bucket_name))
    try:
        files = cos.Bucket(bucket_name).objects.all()
        for file in files:
            print("Item: {0} ({1} bytes).".format(file.key, file.size))
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to retrieve bucket contents: {0}".format(e))
    return files

def get_item(bucket_name, item_name):
    print("Retrieving item from bucket: {0}, key: {1}".format(bucket_name, item_name))
    try:
        file = cos.Object(bucket_name, item_name).get()
        # print("File Contents: {0}".format(file["Body"].read()))
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to retrieve file contents: {0}".format(e))
    return file["Body"].read()

all_files = get_bucket_item_list(BUCKET_NAME)
for file in all_files:
    if file.size > 0:
        file_content = get_item(BUCKET_NAME, file.key)
        # Download this file locally
        d = BASE_PATH + "/" + '/'.join(file.key.split('/')[0:-1])
        if not os.path.exists(d):
            os.makedirs(d)
        f = open(BASE_PATH + "/" + file.key, "wb")
        f.write(file_content)
        f.close()
# taking the rest from here - we need to load the model and run it
# https://www.kaggle.com/abhishek/pytorch-bert-inference

import sys
package_dir = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"
sys.path.append(package_dir)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.utils.data
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import warnings
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
from pytorch_pretrained_bert import BertConfig

warnings.filterwarnings(action='once')
# we have to comment out the below because the notebook is running on kaggle's server which unfortunately doesnt allow for cuda training in its free version
# device = torch.device('cuda')
# instead we define the device as CPU based
device = torch.device('cpu')
def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in tqdm(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    return np.array(all_tokens)
MAX_SEQUENCE_LENGTH = 220
SEED = 1234
BATCH_SIZE = 32
BERT_MODEL_PATH = BASE_PATH + "/" + "vm_output/data/uncased_L-12_H-768_A-12"

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

bert_config = BertConfig(BERT_MODEL_PATH + "/bert_config.json")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)
test_df = pd.read_csv(BASE_PATH + "/vm_output/data/test.csv")
test_df['comment_text'] = test_df['comment_text'].astype(str) 

# Unfortunately, since we are evaluating on a cpu based backend and cuda is disabled, doing inference for 3000+ test batches (when batch size = 32)
# is going to take over 10 hours! Instead, we consider only the top n rows which get processed in batches of 32
test_df = test_df[:128]


X_test = convert_lines(test_df["comment_text"].fillna("DUMMY_VALUE"), MAX_SEQUENCE_LENGTH, tokenizer)
model = BertForSequenceClassification(bert_config, num_labels=1)
model.load_state_dict(torch.load(BASE_PATH + "/vm_output/bert_pytorch.bin", map_location=device))
model.to(device)
for param in model.parameters():
    param.requires_grad = False
model.eval()

test_preds = np.zeros((len(X_test)))
test = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))
test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)
tk0 = tqdm(test_loader)
for i, (x_batch,) in enumerate(tk0):
    pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
    test_preds[i * 32:(i + 1) * 32] = pred[:, 0].detach().cpu().squeeze().numpy()

test_pred = torch.sigmoid(torch.tensor(test_preds)).numpy().ravel()
submission = pd.DataFrame.from_dict({
    'id': test_df['id'],
    'prediction': test_pred
})
submission.to_csv('submission.csv', index=False)
submission[10:30]
