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
