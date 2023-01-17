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
#Prerequisites
!pip install cdqa

#This implementation uses the cdQA-Suite
import os
import pandas as pd
from ast import literal_eval

from cdqa.utils.filters import filter_paragraphs
from cdqa.pipeline import QAPipeline
from cdqa.utils.download import download_model
#It uses a Pre-trained model on SQuAD v2.0
download_model(model='bert-squad_1.1', dir='./models')
!pip install docx2python
from docx2python import docx2python
from shutil import copyfile
copyfile(src = "../input/trump-corona/article_text.docx", dst = "../working/article_text.docx")
doc=docx2python('../input/trump-corona/article_text.docx')
doc.body[0][0][0]
df2={'title':['article'], 'paragraphs':[doc.body[0][0][0]]};
#The cdQA suite requires input to be in a particular form of a Pandas Dataframe
import pandas as pd
data=pd.DataFrame(df2)
cdqa_pipeline = QAPipeline(reader='./models/bert_qa.joblib')
cdqa_pipeline.fit_retriever(df=data)
query = 'How many confirmed cases are there in the US?'
prediction = cdqa_pipeline.predict(query)
print('query: {}'.format(query))
print('answer: {}'.format(prediction[0]))
print('title: {}'.format(prediction[1]))
print('paragraph: {}'.format(prediction[2]))
query2 = 'How many people have died?'
prediction2 = cdqa_pipeline.predict(query2)
print('query: {}'.format(query2))
print('answer: {}'.format(prediction2[0]))
print('title: {}'.format(prediction2[1]))
print('paragraph: {}'.format(prediction2[2]))