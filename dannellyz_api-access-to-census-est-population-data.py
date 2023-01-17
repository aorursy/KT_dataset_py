#Load API key from Kaggle Secret

#Can find at add-on secret at top of notebook

#Can apply for a free API key here: https://msr-apis.portal.azure-api.net/products



from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

census_api_key = user_secrets.get_secret("census_api_key")
import pandas as pd

import requests



r= requests.get("https://api.census.gov/data/2019/pep/population?get=POP,NAME,DENSITY&for=state:*&key={}".format(census_api_key))

results = r.json()

columns = results.pop(0)

pd.DataFrame(results, columns=columns)