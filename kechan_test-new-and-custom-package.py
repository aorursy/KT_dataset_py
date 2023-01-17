!pip3 list |grep pandas
%pip install -U pip 

%pip install pandas==0.24.0
!pip3 list |grep pandas
import pandas as pd

pd.__version__
idx = pd.period_range('2000', periods=4)

idx.array
from google.cloud import bigquery
client = bigquery.Client()
hn_dataset_ref = client.dataset('hacker_news', project='bigquery-public-data')
type(hn_dataset_ref)
hn_dset = client.get_dataset(hn_dataset_ref)