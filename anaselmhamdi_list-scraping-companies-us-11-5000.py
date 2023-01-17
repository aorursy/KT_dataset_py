# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/free-7-million-company-dataset/companies_sorted.csv')
df['size range'].unique().tolist()
sum(df['size range'].isnull())/len(df)
target_range = ['1001 - 5000',
 '501 - 1000',
 '201 - 500',
 '51 - 200',
 '11 - 50']
dfilter = df[(df.industry.str.contains('technology and services')) & (df.country == 'united states') & (df['size range'].isin(target_range))]
dfilter
sum(df.domain.isnull())/len(df)
import requests
api_key = '690bf60971ecd0bac14b079ae059094e89fe9883' # its a free one i dont care about hiding it
def hunter_enrich(domain):
    if domain:
        try:
            r = requests.get(f"https://api.hunter.io/v2/domain-search?domain={domain}&api_key={api_key}")
            return r.json()['data']['emails']
        except:
            return []
    else:
        return []
domains = dfilter.domains.tolist() # gets all the domains
domains_enriched = [{'domain': domain, 'people':hunter_enrich(domain)} for domain in domains] # enriches all the domains
# mergedDf = dfilter.join(on='domain', how='left') # merges back the data together
# # Sample output from function hunter_enrich : 

# [
#       {
#         "value": "daniel@stripe.com",
#         "type": "personal",
#         "confidence": 99,
#         "sources": [
#           {
#             "domain": "stripe.com",
#             "uri": "http://stripe.com/events/rubykaigi-2019-breakfast",
#             "extracted_on": "2019-07-14",
#             "last_seen_on": "2020-08-11",
#             "still_on_page": true
#           },
#           {
#             "domain": "stripe.com",
#             "uri": "http://stripe.com/events/google-jp-nov-2017",
#             "extracted_on": "2018-10-28",
#             "last_seen_on": "2020-07-26",
#             "still_on_page": true
#           }
#         ],
#         "first_name": "Daniel",
#         "last_name": "Heffernan",
#         "position": "Product Manager",
#         "seniority": "senior",
#         "department": "management",
#         "linkedin": null,
#         "twitter": "danielshi",
#         "phone_number": null,
#         "verification": {
#           "date": "2020-07-07",
#           "status": "valid"
#         }
#       },
#       {
#         "value": "hamish.kerr@stripe.com",
#         "type": "personal",
#         "confidence": 99,
#         "sources": [
#           {
#             "domain": "stripe.com",
#             "uri": "http://stripe.com/blog/connect-dashboard-updates-june-2018",
#             "extracted_on": "2020-06-17",
#             "last_seen_on": "2020-09-06",
#             "still_on_page": true
#           },
#           {
#             "domain": "stripe.com",
#             "uri": "http://stripe.com/es-ch/blog/connect-dashboard-updates-june-2018",
#             "extracted_on": "2020-04-15",
#             "last_seen_on": "2020-07-21",
#             "still_on_page": true
#           },
#           {
#             "domain": "stripe.com",
#             "uri": "http://stripe.com/es-ee/blog/connect-dashboard-updates-june-2018",
#             "extracted_on": "2020-04-15",
#             "last_seen_on": "2020-07-15",
#             "still_on_page": true
#           },
#           {
#             "domain": "stripe.com",
#             "uri": "http://stripe.com/it-au/blog/connect-dashboard-updates-june-2018",
#             "extracted_on": "2020-04-15",
#             "last_seen_on": "2020-07-16",
#             "still_on_page": true
#           },
#           {
#             "domain": "stripe.com",
#             "uri": "http://stripe.com/blog/page/6",
#             "extracted_on": "2020-05-15",
#             "last_seen_on": "2020-05-15",
#             "still_on_page": false
#           },
#           {
#             "domain": "uploads.stripe.com",
#             "uri": "http://uploads.stripe.com/blog/page/3",
#             "extracted_on": "2019-05-03",
#             "last_seen_on": "2019-05-25",
#             "still_on_page": false
#           },
#           {
#             "domain": "uploads.stripe.com",
#             "uri": "http://uploads.stripe.com/blog/page/2",
#             "extracted_on": "2019-01-23",
#             "last_seen_on": "2019-05-02",
#             "still_on_page": false
#           },
#           {
#             "domain": "uploads.stripe.com",
#             "uri": "http://uploads.stripe.com/blog",
#             "extracted_on": "2018-07-31",
#             "last_seen_on": "2018-09-17",
#             "still_on_page": false
#           },
#           {
#             "domain": "uploads.stripe.com",
#             "uri": "http://uploads.stripe.com/blog/page/1",
#             "extracted_on": "2018-06-21",
#             "last_seen_on": "2018-09-21",
#             "still_on_page": false
#           }
#         ],
#         "first_name": "Hamish",
#         "last_name": "Kerr",
#         "position": "Resource Management",
#         "seniority": null,
#         "department": null,
#         "linkedin": null,
#         "twitter": "hamish_kerr",
#         "phone_number": null,
#         "verification": {
#           "date": null,
#           "status": null
#         }
#       }
# ]