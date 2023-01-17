import hashlib

import os

import pathlib

import platform

import sys

import warnings



import pandas as pd



from tqdm import tqdm
# Filter warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings('ignore')
# Get working directory

try:

    path_working_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

except:

    path_working_dir = os.path.abspath(str(pathlib.Path().resolve()))
# Set input directory

path_input  = '../input/rsna-intracranial-hemorrhage-detection'

path_output = './'

assert os.path.exists(path_input) == True
# Calc hashsum of files with selected file extension

list_output = []

list_ext    = ['.csv', '.dcm']

for (dirpath, dirnames, filenames) in os.walk(path_input):

    print(dirpath.replace(path_input, '.'), file=sys.stderr)

    for filename in tqdm(filenames):

        if max([filename.find(ext) for ext in list_ext]) > -1:

            with open(os.path.join(dirpath, filename), 'rb') as fp:

                fp_read = fp.read()

                list_output.append([os.path.join(dirpath.replace(path_input, '.'), filename), hashlib.md5(fp_read).hexdigest(), hashlib.sha1(fp_read).hexdigest()])
# Output result

df_output = pd.DataFrame(list_output, columns=['filename', 'md5', 'sha1'])

df_output = df_output.sort_values(by='filename')

df_output.to_csv(os.path.join(path_output, 'df_output.csv'), index=False)
df_output.head()
!md5sum df_output.csv
!sha1sum df_output.csv