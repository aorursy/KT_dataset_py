# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import zipfile

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import requests

import smart_open

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import requests

import smart_open

import random

def opus_mono_textgen(lang='fr',chunk_size=int(32e6)):

    r=requests.get("http://opus.nlpl.eu/opusapi/",params={"source":"fr","version":"latest", 'preprocessing': 'mono'})

    d=r.json()

    urls=sorted(set( c['url']   for c in d['corpora'] if  'fr.txt.gz' in c['url'] and int(c['size'])>5e3 ))

    random.shuffle(urls)

    for u in urls:

        try:

            with smart_open.open(u) as f:  

              t=f.read(chunk_size)

              while len(t)>0:

                yield t

                t=f.read(1024)

        except:

            pass

        



print(next(opus_mono_textgen(lang='fr',chunk_size=800)))
with   zipfile.ZipFile("opus_fr.zip", mode="w") as zf:

    n=1

    for t in opus_mono_textgen(lang='fr',chunk_size=int(32e6)):

        zf.writestr(f"text{n}.txt",t)

        n+=1

        st=os.statvfs(".")

        disk_free=st.f_bavail*st.f_bsize

        if n%1000==0:

            print(n,disk_free,t[:800])

        if disk_free<512*1024:

            break   