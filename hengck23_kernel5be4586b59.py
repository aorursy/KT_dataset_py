# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#  Kaggle Other PA Funded Credit elizabethpark 43693411



#import os

#os.environ['GOOGLE_APPLICATION_CREDENTIALS']='/kaggle/input/automl/authorise/kaggle-bengaliai-4228c6b24a8c.json'



def set_credential_from_json_string():

    import json

    # !pip install google-cloud-pubsub

    from google.cloud import pubsub 

    

    text = {

      "type": "service_account",

      "project_id": "kaggle-bengaliai",

      "private_key_id": "<HIDDEN XXXXXXX>",

      "private_key": "-----BEGIN PRIVATE KEY-----\n <HIDDEN XXXXXXX>", 

        "client_email": "local-dev@kaggle-bengaliai.iam.gserviceaccount.com",

      "client_id": "117266300789064494401",

      "auth_uri": "https://accounts.google.com/o/oauth2/auth",

      "token_uri": "https://oauth2.googleapis.com/token",

      "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",

      "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/local-dev%40kaggle-bengaliai.iam.gserviceaccount.com"

    }

    credential = json.dumps(text)

    return credential





from google.cloud import automl

#from google.cloud import automl_v1beta1 as automl



PROJECT_ID    = 'kaggle-bengaliai' 



#this is ok

#automl_client = automl.AutoMlClient(credentials=set_credential_from_json_string())



#this fails

automl_client = automl.AutoMlClient()



print('sucess')