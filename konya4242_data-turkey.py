# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df1 = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')

df2 = pd.read_csv('../input/kaggle-survey-2019/questions_only.csv')

df3 = pd.read_csv('../input/kaggle-survey-2019/survey_schema.csv')

df4 = pd.read_csv('../input/kaggle-survey-2019/other_text_responses.csv')
a=df1["Q3"].value_counts()

b=a.index

plt.figure(figsize=(20,20))

sns.barplot(x=a,y=b)

plt.show()
df1.head()
aa=df1[df1["Q3"]=="Turkey"] 

prog=aa["Q34_Part_4"].unique()

prog
meslek=aa["Q5"].value_counts()

meslek
erkek_öğrenci=df1[(df1["Q3"]=="Turkey")&(df1["Q5"]=="Student")&(df1["Q2"]=="Male")] 

erkek_öğrenci


