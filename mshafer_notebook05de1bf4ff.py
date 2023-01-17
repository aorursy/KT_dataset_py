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
questions_df = pd.read_csv("/kaggle/input/stacksample/Questions.csv", encoding="ISO-8859-1")

questions_df.head(5)
def count_word_in_questions(word, column):

    return sum([1 if word.lower() in item.lower() else 0 for item in questions_df[column]])

count_word_in_questions('flask', 'Title') + count_word_in_questions('flask', 'Body')
count_word_in_questions('fastapi', 'Title') + count_word_in_questions('fastapi', 'Body')
count_word_in_questions('fast api', 'Title') + count_word_in_questions('fast api', 'Body')
answers_df = pd.read_csv("/kaggle/input/stacksample/Answers.csv", encoding="ISO-8859-1")

answers_df.head(5)
def count_word_in_answers(word, column):

    return sum([1 if word.lower() in item.lower() else 0 for item in answers_df[column]])
count_word_in_answers('flask', 'Body')
count_word_in_answers('fastapi', 'Body') + count_word_in_answers('fast api', 'Body')