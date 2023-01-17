# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def df_1(sheet_name=None):

    df_1 = pd.read_excel('/kaggle/input/examresult/py_mind.xlsx', sheet_name=sheet_name)

    return df_1

def df_2(sheet_name=None):

    df_2 = pd.read_excel('/kaggle/input/examresult/py_opinion.xlsx', sheet_name=sheet_name)

    return df_2

def df_3(sheet_name=None):

    df_3 = pd.read_excel('/kaggle/input/examresult/py_science.xlsx', sheet_name=sheet_name)

    return df_3

def df_4(sheet_name=None):

    df_4 = pd.read_excel('/kaggle/input/examresult/py_sense.xlsx', sheet_name=sheet_name)

    return df_4
student_names_1 = [list(df_1().items())[i][0] for i in range(len(df_1()))]

student_names_2 = [list(df_2().items())[i][0] for i in range(len(df_2()))]

student_names_3 = [list(df_3().items())[i][0] for i in range(len(df_3()))]

student_names_4 = [list(df_4().items())[i][0] for i in range(len(df_4()))]
print(student_names_1,student_names_2,student_names_3,student_names_4,sep="\n\n")
answer_key={}

question_number=1



for i in range(20):

    answer_key[question_number]=df_1('emrullah').loc[i,'Cevap A.']

    question_number+=1

print(answer_key)


student_answers_1 = pd.DataFrame(dict(zip(student_names_1,[df_1(sheet_name=name).loc[:,'ogr.C'] for name in student_names_1])))

print(student_answers_1)

a= pd.DataFrame(dict(zip(student_names_1,[df_1(sheet_name=name).loc[:,'ogr.C'] for name in student_names_2])))