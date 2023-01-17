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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

#import seaborn as sns

from sklearn.model_selection import train_test_split

from wordcloud import WordCloud

import seaborn as sns

import re
df =pd.read_csv('/kaggle/input/fake-news/train.csv' ,  encoding='ISO-8859-1')



df.head()
df.isna().sum()
df_news = df[~df['text'].isna()][['text','label']]
df_news.info()
df_news.drop_duplicates(subset=['text'],keep='first',inplace=True)

df_news.info()
fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(17, 5), dpi=100)

length=df_news[df_news["label"]==1]['text'].str.len()

ax1.hist(length,bins = 20,color='skyblue')

ax1.set_title('Fake News')

length=df_news[df_news["label"]==0]['text'].str.len()

ax2.hist(length, bins = 20)

ax2.set_title('Real News')

fig.suptitle('Characters in text')

plt.show()
text = " ".join([x for x in df_news.text])



wordcloud = WordCloud(background_color='white').generate(text)



plt.figure(figsize=(8,6))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis('off')

plt.show()
## for fake



text = " ".join([x for x in df_news.text[df_news.label==1]])



wordcloud = WordCloud(background_color='white').generate(text)



plt.figure(figsize=(8,6))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis('off')

plt.show()
## for real



text = " ".join([x for x in df_news.text[df_news.label==0]])



wordcloud = WordCloud(background_color='white').generate(text)



plt.figure(figsize=(8,6))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis('off')

plt.show()
print('Number of 0 (Not Fake) : ', df_news["label"].value_counts()[0])

print('Number of 1 (Fake) : ', df_news["label"].value_counts()[1])
label = df_news["label"].value_counts()

sns.barplot(label.index, label)

plt.title('Target Count', fontsize=14)
# Dataset Preprocessing

def text_cleaning(text):

    text = re.sub("[^a-zA-Z]", " ", text) # removing punctuation

    return text



df_news['text'] = df_news['text'].apply(text_cleaning)
train_df,eval_df = train_test_split(df_news,test_size = 0.05)
!pip install simpletransformers==0.32.3
from simpletransformers.classification import ClassificationModel





# Create a TransformerModel

model = ClassificationModel('bert', 'bert-base-cased', num_labels=2, 

                            args={'reprocess_input_data': True, 'overwrite_output_dir': True},use_cuda=False)

model.train_model(train_df)
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
print(result)

print(model_outputs)
lst = []

for arr in model_outputs:

    lst.append(np.argmax(arr))
true = eval_df['label'].tolist()

predicted = lst
import sklearn

mat = sklearn.metrics.confusion_matrix(true , predicted)

mat
print(sklearn.metrics.classification_report(true,predicted,target_names=['real','fake']))

test_df =pd.read_csv('/kaggle/input/fake-news/test.csv' ,  encoding='ISO-8859-1')



test_df.head()
test_df.isna().sum()
test_df.fillna('' , inplace=True)
test_df['text'] = test_df['text'].apply(text_cleaning)
final_prediction = model.predict(list(test_df.text))
final_prediction
print('Loading in Submission File...')



submit_df = pd.read_csv("/kaggle/input/fake-news/submit.csv")

print(submit_df.columns)

submit_df['label'] = final_prediction[0]



submit_df.to_csv('bert_submit.csv', index=False)