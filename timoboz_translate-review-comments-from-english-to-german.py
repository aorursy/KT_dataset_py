# Install packages

!pip install --upgrade google-cloud-translate==2.0.0

!pip install --upgrade google-auth
# Handle credentials

import json

from google.oauth2 import service_account

from kaggle_secrets import UserSecretsClient



user_secrets = UserSecretsClient()

secret_value = user_secrets.get_secret("translation-playground")



service_account_info = json.loads(secret_value)

credentials = service_account.Credentials.from_service_account_info(

    service_account_info)
# Setup client & translation function

from google.cloud import translate_v2 as translate



translate_client = translate.Client(credentials=credentials)



def translate(text, target_lang, source_lang="en"):

    try:    

        result = translate_client.translate(text, target_language=target_lang, source_language=source_lang)

        return result['translatedText']

    except:

        return ""
# Test it

print(translate("This is a very nice text to translate", "de"))

import pandas as pd



data = pd.read_csv("../input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv")

data.head()



#data = data[:10]
data['Review Text DE'] = data.apply(lambda row: translate(row['Review Text'], "de"), axis = 1)



data.head()
data.to_csv("/kaggle/working/Reviews_DE.csv");