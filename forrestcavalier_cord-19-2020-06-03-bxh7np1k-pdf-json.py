import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json



cord19_path = "/kaggle/input/CORD-19-research-challenge/"



df = pd.read_csv(cord19_path + "metadata.csv", index_col = "cord_uid", dtype = str, na_filter= False)



uidlist = ['bxh7np1k', 'a0hn3ob2', 'f3jmr4se', 'ipwox31c', 'iq8ior67']

metadata = df.loc[uidlist]



for i in range(0,len(uidlist)):

    thisRow = metadata.iloc[i];

    print('\ncord_uid=' + uidlist[i])

    print('Title: ' + thisRow.title)

    print(thisRow.url)

    print(thisRow.pdf_json_files)

    if (thisRow.pdf_json_files != ''):

        file_path = cord19_path + thisRow.pdf_json_files

        with open(file_path) as json_file:

           jf = json.load(json_file)

        print('body_text[0].text=' + jf['body_text'][0]['text'])




