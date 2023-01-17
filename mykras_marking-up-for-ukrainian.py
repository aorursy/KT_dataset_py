!pip install tokenize_uk
!pip install utils
!pip install colorama
!pip install pigeon-jupyter
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

from utils import *
import tokenize_uk
from colorama import Fore, Back, Style

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
files = ['/kaggle/input/ukrainian-texts/Lys_mykyta.txt']
files
for src_file in files:
    with open(src_file, 'rb') as f:
        data = f.read()
    text = data.decode('utf-8')
    tokens_text = tokenize_uk.tokenize_sents(text)
tokens_text
len(tokens_text)
tokens_text[:5]
#
from pigeon import annotate
annotations = annotate(
tokens_text[:25],
options=['улюбленний_фрагмент', 'нейтрально']
)
df = pd.DataFrame(annotations, columns=['uk_texts', 'target'])
df
def clean_text(text):
  text = str(text)
  text = re.sub(r'[0-9"]', '', text)
  text = re.sub(r'#[\S]+\b', '', text)
  text = re.sub(r'@[\S]+\b', '', text)
  text = re.sub(r'https?\S+', '', text)
  text = re.sub(r'\s+', ' ', text)
  return text.strip()
df['uk_texts'] = df['uk_texts'].apply(lambda x: clean_text(x))
df