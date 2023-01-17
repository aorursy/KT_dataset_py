import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

current_sensor_data_type = pd.read_csv("../input/feeddata/feed.csv")


current_sensor_data_type.isnull().values.any()

current_sensor_data_type.shape

current_sensor_data_type.head()



import seaborn as sns

sns.countplot(x='field2', data=current_sensor_data_type)

y = current_sensor_data_type['field2']
y = np.array(list(map(lambda x: 1 if x>0 else 0, y))) 










