# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/songdata.csv')
df.head()
!git clone https://github.com/jsvine/markovify
import markovify as mk
mk_models = {}
import re
pattern = re.compile('\n *\n')
data = {}
for index, row in df.iterrows():
    artist = row['artist']
    text = '\n'.join([t.strip().lower() for t in pattern.split(row['text'])])
    if artist not in data:
        data[artist] = text + '\n'
    else:
        data[artist] = data[artist] + text
data['ABBA']
#markovify
for k,v in data.items():
    mk_models[k] = mk.NewlineText(v)
for i in range(5):
    print(mk_models['The Beatles'].make_sentence())
for i in range(5):
    print(mk_models['Queen'].make_sentence(max_overlap_ratio=.80))
mk_models.keys()
for i in range(5):
    print(mk_models['Kanye West'].make_sentence(max_overlap_ratio=.80))
for i in range(5):
    print(mk_models['Marilyn Manson'].make_sentence(max_overlap_ratio=.80))
for i in range(5):
    print(mk_models['Weird Al Yankovic'].make_sentence(max_overlap_ratio=.80))
for i in range(5):
    print(mk_models['Marilyn Manson'].make_sentence(max_overlap_ratio=.80))
print("")
for i in range(5):
    print(mk_models['Marilyn Manson'].make_sentence(max_overlap_ratio=.80))
for i in range(5):
    print(mk_models['Radiohead'].make_sentence(max_overlap_ratio=.80))
print("")
for i in range(5):
    print(mk_models['Radiohead'].make_sentence(max_overlap_ratio=.80))
for i in range(5):
    print(mk_models['Radiohead'].make_sentence(max_overlap_ratio=.80))