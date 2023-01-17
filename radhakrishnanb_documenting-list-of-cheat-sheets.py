!pip install PyPDF2

import numpy as np

import pandas as pd

import os

from PyPDF2 import PdfFileReader



def get_pages(file_path):

    try:

        if file_path[-4:] == '.pdf':

            pdf = PdfFileReader(open(file_path,'rb'),strict=False)

            return(int(pdf.getNumPages()))

    except:

        pass

    return np.nan



def add_files():

    file_paths = []

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            current_path = os.path.join(dirname, filename)

            file_paths.append(current_path)

    return sorted(file_paths)



def get_df():

    df = pd.DataFrame()

    df['file_paths'] = add_files()

    df['modified_paths'] = df['file_paths'].map(lambda x: x.replace('/kaggle/input/data-science-cheat-sheets/', ''))

    df['extensions']  = df['file_paths'].map(lambda x: x.split('.')[-1])

    df = df[df.extensions.map(lambda x: x not in ['md', 'ini', 'tex'])]

    df['folders'] =  df['modified_paths'].map(lambda x: x.split('/')[0] if len(x.split('/'))>1 else '' )

    df['MB size'] = df['file_paths'].map(lambda x: round(os.stat(x).st_size/1024**2,2))

    df['pages'] =df['file_paths'].apply(get_pages)

    return df



def show_cheat_sheet(df):

    df = df.copy()

    df['cheat'] = df['modified_paths'].map(lambda x: 'cheat' in x.lower().split('/')[-1])

    df['sheet'] = df['modified_paths'].map(lambda x: 'sheet' in x.lower().split('/')[-1])

    return df[df['cheat'] & df['sheet']].drop(columns=['cheat', 'sheet']).reset_index(drop=True)

original_file_list = get_df()

df= show_cheat_sheet(original_file_list)

df.drop(columns=['file_paths'])
list(df['folders'].unique())
from wand.image import Image as WImage

def display_cheatsheet(index, df=df):

    data = df[df.index == index].to_dict()

    path = list(data['file_paths'].values())[0].replace('/kaggle/input/', '../input/')

    return WImage(filename=path)
display_cheatsheet(11)
original_file_list.drop(columns=['file_paths']).head()
list(original_file_list['folders'].unique())
def get_files_in_folder(folder_name):

    return original_file_list[original_file_list['folders']==folder_name].drop('file_paths', axis=1)



get_files_in_folder('NLP')
display_cheatsheet(179, original_file_list)