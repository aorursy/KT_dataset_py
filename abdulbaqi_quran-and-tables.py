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
!pip install datascience
from datascience import *
q = Table.read_table('/kaggle/input/quran-clean-without-araab/Quran-clean-without-aarab.csv')
q
q.show(3)
q.select('Ayah')
q.num_rows
q.row(q.num_rows-1)
def count_words(item):

    return len(item.split())
q.apply(count_words, 'Ayah')
qwc = q.with_columns("words", q.apply(count_words, 'Ayah'))

qwc
qwc = qwc.relabeled(0,'SrNo')

qwc
qwc.row(qwc.num_rows-1)
en = Table.read_table('/kaggle/input/qurancsv/Quran.csv')

en
en = en.relabeled(4, 'en')

en
en = en.select(['SrNo', 'en'])

en
en = en.with_columns('sr', en.column('SrNo')-1).drop('SrNo')

en
en.row(en.num_rows-1)
quran = qwc.join('SrNo', en, 'sr')

quran
quran = quran.relabeled(1, 'sno').relabeled(2,'vno')

quran
quran.where('sno', 2).where('vno',255)
quran.where('vno', 1)
def discount_basmalah(sura_no, verse_no, words):

    if (sura_no==1 or sura_no==9):

        return words

    if (verse_no == 1):

        return words-4

    else:

        return words
quran = quran.with_columns('wc', quran.apply(discount_basmalah,'sno','vno', 'words'))

quran
q_place = Table.read_table('/kaggle/input/quran-makki-madani/quran-toc.csv')

q_place
q_select = q_place.select(['No.','Place'])

q_select
quran = quran.join('sno',q_select,'No.')

quran
quran.sort('wc', descending=True)
quran.where('Place', 'Meccan').sort('wc', descending=True)
quran2 = quran.drop('words')

quran2
lc = quran2.apply(len, 'Ayah')-quran2.column('wc')+1
quran2 = quran2.with_columns('lc', lc)

quran2
quran2.sort('lc', descending=True)
np.array(quran2.column('lc')).sum()/np.array(quran2.column('wc')).sum()
qmeccan = quran2.where('Place','Meccan').select('wc','lc')

np.array(qmeccan.column('lc')).sum()/np.array(qmeccan.column('wc')).sum()
qmedinan = quran2.where('Place','Medinan').select('wc','lc')

np.array(qmedinan.column('lc')).sum()/np.array(qmedinan.column('wc')).sum()
quran.to_csv('quran-en-ar-place.csv')