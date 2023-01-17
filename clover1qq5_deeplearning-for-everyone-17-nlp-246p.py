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
text = '해보지 않으면 해낼 수 없다'
result = text_to_word_sequence(text)

print("\n원문:\n", text)

print("\n토큰화:\n", result)
from keras.preprocessing.text import Tokenizer
docs = ['먼저 텍스트의 각 단어를 나누어 토큰화 합니다.', 

        '텍스트의 단어로 토큰화 해야 딥러닝에서 인식됩니다.',

        '토큰화 한 결과는 딥러닝에서 사용할 수 있습니다.',

        ]



token = Tokenizer()

token.fit_on_texts(docs)
print("\n문장 카운트:", token.document_count)

print("\n각 단어가 몇 개의 문장에 포함되어 있는가:\n", token.word_docs)

print("\n각 단어에 매겨진 인덱스 값:\n", token.word_index)