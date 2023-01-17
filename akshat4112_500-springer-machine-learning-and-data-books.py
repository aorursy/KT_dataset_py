!pip install wget



import requests, wget

import pandas as pd
df = pd.read_excel("/kaggle/input/FreeEnglishtextbooks.xlsx")
for index, row in df.iterrows():

        # loop through the excel list

        file_name = f"{row.loc['Book Title']}_{row.loc['Edition']}".replace('/','-').replace(':','-')

        url = f"{row.loc['OpenURL']}"

        r = requests.get(url) 

        download_url = f"{r.url.replace('book','content/pdf')}.pdf"

        wget.download(download_url, f"{file_name}.pdf") 

        print(f"downloading {file_name}.pdf Complete ....")