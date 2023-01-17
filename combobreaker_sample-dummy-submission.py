import pandas as pd



lenta = pd.read_csv('../input/lenta-ru-ozon-2020/lenta-ru-train.csv')

lenta.head()
lenta.topic.value_counts()
len(lenta)
from random import randint



test_data = pd.read_csv('../input/lenta-ru-ozon-2020/lenta-ru-test.csv')

test_data.head()
test_data['category'] = [randint(0,4) for i in range(len(test_data)) ] # тут должен быть ваш предсказанный таргет

test_data['id'] = test_data.index

test_data.head()
test_data[['id','category']].to_csv('dummy_submission.csv', index=False)