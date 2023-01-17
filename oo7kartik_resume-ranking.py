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
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

import json



cv = CountVectorizer()



resume_list = []

job_list = []



path = '/kaggle/input'



for i in range(100):

    with open(path + '/resumes_text/resume{}.txt'.format(i+1), 'r') as file:

        resume_list.append(file.read())



job_desc_number = 3

for i in range(job_desc_number, job_desc_number + 1):

    with open(path + '/job_desc_dir/job_desc{}.txt'.format(i), "r") as file:

        job_list.append(file.read())



ranking_list = []



for i, job_desc in enumerate(job_list):

    ranking_list.append([])

    for j, resume in enumerate(resume_list):

        text = [resume, job_desc]



        count_matrix = cv.fit_transform(text)

        ranking_list[i].append([j, cosine_similarity(count_matrix)[0][1] * 100])



def func(lis):

    lis = sorted(lis, key = lambda x : -x[1])

    return [[x[0]+1, x[1]] for x in lis]





ranking_list = list(map(func, ranking_list))



for elem in ranking_list[0]:

    print(elem)
