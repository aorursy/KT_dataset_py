import pandas as pd

import os

os.listdir('../input/')
PATH_TO_PRIMARY_DATA = '../input/mlcourse-dota2-win-prediction'

PATH_TO_ADDITIONAL_DATA = '../input/source-kernel-example'

df_train = pd.read_csv(os.path.join(PATH_TO_PRIMARY_DATA, 'train_features.csv'), index_col='match_id_hash')

df_train.head(2)
df_example= pd.read_csv(os.path.join(PATH_TO_ADDITIONAL_DATA, 'output.csv'), index_col='Unnamed: 0')

df_example.head(2)
from IPython.display import HTML

def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)
df_example.to_csv('example.csv')

create_download_link(filename='example.csv')