# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
fake_news=pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')
real_news=pd.read_csv('../input/fake-and-real-news-dataset/True.csv')

fake_news


real_news

fake_news.shape
real_news.shape
fake_news.columns

real_news.columns
real_news['True_or_False']=1
fake_news['True_or_False']=0


real_news.head()
fake_news.head()

fake_news.info()


real_news.info()


real_news.describe()


fake_news.describe()
fake_news_subject=fake_news.subject
fake_news_subject.unique()

real_news_subject=real_news.subject
real_news_subject.unique()
print(real_news.subject.value_counts(dropna=False))
print(fake_news.subject.value_counts(dropna=False))