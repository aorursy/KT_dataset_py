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
import json

import plotly.express as px

import seaborn as sns

from matplotlib import pyplot as plt

from nltk.tokenize import wordpunct_tokenize

from tqdm import tqdm
train = pd.read_csv('/kaggle/input/deepnlp-hse-course/train.csv')
train.head()
# читаем 

with open('/kaggle/input/deepnlp-hse-course/main_category_mapper.json') as f:

    main_cat2id = json.load(f)

    

# инвертируем

id2main_cat = {value: key for key, value in main_cat2id.items()}
id2main_cat
# тоже самое с доп категориями

with open('/kaggle/input/deepnlp-hse-course/sub_category_mapper.json') as f:

    sub_cat2id = json.load(f)

    

id2sub_cat = {value: key for key, value in sub_cat2id.items()}
# переводим id в строку

train.main_category = train.main_category.map(id2main_cat)

train.sub_category = train.sub_category.map(id2sub_cat)
train.head()
val_counts_main = pd.DataFrame(train.main_category.value_counts())

val_counts_main.reset_index(inplace=True)

val_counts_main.columns = ['category', 'n_entries']

fig = px.bar(val_counts_main, x='category', y='n_entries')

fig.show()
val_counts_sub = pd.DataFrame(train.sub_category.value_counts())

val_counts_sub.reset_index(inplace=True)

val_counts_sub.columns = ['category', 'n_entries']

fig = px.bar(val_counts_sub, x='category', y='n_entries')

fig.show()
train['char_len'] = train.question.map(len)
train['token_len'] = train.question.map(lambda x: len(wordpunct_tokenize(x)))
train.head()
plt.figure(figsize=(16, 12))

plt.title('Distplot question char len')

sns.distplot(train.char_len)
plt.figure(figsize=(16, 12))

plt.title('Distplot question token len')

sns.distplot(train.token_len)
input_file = open('/kaggle/input/deepnlp-hse-course/answers.jsonl')



progress_bar = tqdm()



additional_data = {

    'question': [],

    'answer': []

}



try:

    while True:



        line = input_file.readline().strip()



        if not line:

            break

            

        line = json.loads(line)

        

        question = list(line.keys())[0]

        

        for answer in line[question]:

            additional_data['question'].append(question)

            additional_data['answer'].append(answer)

            

        progress_bar.update()

            

except KeyboardInterrupt:

    pass



progress_bar.close()

input_file.close()
additional_data = pd.DataFrame(data=additional_data)
additional_data.sample(frac=1).head(10)
f'Количество строк в таблице: {additional_data.shape[0]}'
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score, confusion_matrix
# переводим класс в индекс

train['target'] = train.main_category.map(main_cat2id)
train.head()
x_train, x_test, y_train, y_test = train_test_split(train.question, train.target, test_size=0.15)
x_test.shape
vectorizer = TfidfVectorizer(max_features=75000, ngram_range=(1, 2))
x_train_vectorized = vectorizer.fit_transform(x_train)

x_test_vectorized = vectorizer.transform(x_test)
log_reg = LogisticRegression(solver='lbfgs', multi_class='auto')
log_reg.fit(x_train_vectorized, y_train)
predicted_train = log_reg.predict(x_train_vectorized)

predicted_test = log_reg.predict(x_test_vectorized)
f1_train = f1_score(y_true=y_train, y_pred=predicted_train, average='micro')

f1_test = f1_score(y_true=y_test, y_pred=predicted_test, average='micro')



f'F1 train: {f1_train:.3f} | test: {f1_test:.3f}'
classes = [id2main_cat[n] for n in range(len(id2main_cat))]
def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=True,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    fig, ax = plt.subplots(figsize=(18, 18))

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax
plot_confusion_matrix(y_test, predicted_test, classes)