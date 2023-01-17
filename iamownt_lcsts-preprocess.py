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
!mkdir /kaggle/working/LCSTS_DATA

!cp /kaggle/input/lcsts-dataset/PART_I.txt /kaggle/working/LCSTS_DATA
file_path = '/kaggle/input/lcsts-dataset/PART_I.txt'

file_path2 = '/kaggle/working/LCSTS_DATA/PART_I.txt'

def show_wrongdata(file_path):

    flag = 0

    for count, line in enumerate(open(file_path,'rU')):

        if count>=2081211*8:

            print(line)

            flag+=1

            if flag>=16:

                break

show_wrongdata(file_path)
def replace_wrongdata(file_path, file_path2):

    p = open(file_path, 'r')

    q = open(file_path2, 'w')

    count = 0

    for i in p:

        count += 1

        if count==(2081211*8+3):

            i='        #M.A.C#圣诞限量彩妆系列\n'

        q.write(i)

    p.close()

    q.close()

replace_wrongdata(file_path, file_path2)

show_wrongdata(file_path2)

from bs4 import BeautifulSoup

import os

from tqdm import tqdm



INPUT = {

    'valid': '/kaggle/input/lcsts-dataset/PART_II.txt',

    'test': '/kaggle/input/lcsts-dataset/PART_III.txt',

}



OUTPUT_DIR = '/kaggle/working/LCSTS_DATA'



qualify = {'valid':0, 'test':0}

unqualify = {'valid':0, 'test':0}

for key in INPUT:

    print('start process: {}\n'.format(key))

    src_file = open(os.path.join(OUTPUT_DIR, key + '.src'), 'a+', encoding='utf-8')

    tgt_file = open(os.path.join(OUTPUT_DIR, key + '.tgt'), 'a+', encoding='utf-8')



    input_file_path = INPUT[key]

    with open(input_file_path, encoding='utf-8') as file:

        contents = file.read()

        soup=BeautifulSoup(contents,'html.parser')

        for doc in tqdm(soup.find_all('doc')):

            short_text = doc.find('short_text').get_text()

            summary = doc.find('summary').get_text()

            human_label = doc.find('human_label').get_text()

            if int(human_label)<=2:

                unqualify[key] += 1

            else:

                 qualify[key] += 1

                 src_file.write(short_text.strip() + '\n')

                 tgt_file.write(summary.strip() + '\n')



    src_file.close()

    tgt_file.close()
INPUT = {

    'train': '/kaggle/working/LCSTS_DATA/PART_I.txt',

}

for key in INPUT:

    print('start process: {}\n'.format(key))

    src_file = open(os.path.join(OUTPUT_DIR, key + '.src'), 'a+', encoding='utf-8')

    tgt_file = open(os.path.join(OUTPUT_DIR, key + '.tgt'), 'a+', encoding='utf-8')



    input_file_path = INPUT[key]

    with open(input_file_path, encoding='utf-8') as file:

        contents = file.read()

        soup=BeautifulSoup(contents,'html.parser')

        for doc in tqdm(soup.find_all('doc')):

            short_text = doc.find('short_text').get_text()

            summary = doc.find('summary').get_text()

            src_file.write(short_text.strip() + '\n')

            tgt_file.write(summary.strip() + '\n')



    src_file.close()

    tgt_file.close()
!rm /kaggle/working/LCSTS_DATA/PART_I.txt

print(unqualify)

print(qualify)