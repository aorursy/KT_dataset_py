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

df =pd.read_csv('/kaggle/input/nlp-getting-started/train.csv' ,  encoding='ISO-8859-1')



df.head()
df_tweets = df[['text','target']]
df_tweets.drop_duplicates(subset=['text'],keep='first',inplace=True)

df_tweets.info()
text = " ".join([x for x in df.text])



wordcloud = WordCloud(background_color='white').generate(text)



plt.figure(figsize=(8,6))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis('off')

plt.show()
## for real



text = " ".join([x for x in df.text[df.target==1]])



wordcloud = WordCloud(background_color='white').generate(text)



plt.figure(figsize=(8,6))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis('off')

plt.show()
## for fake



text = " ".join([x for x in df.text[df.target==0]])



wordcloud = WordCloud(background_color='white').generate(text)



plt.figure(figsize=(8,6))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis('off')

plt.show()
train_df,eval_df = train_test_split(df_tweets,test_size = 0.01)
#pip install transformers==2.10.0
!pip install simpletransformers==0.32.3

from simpletransformers.classification import ClassificationModel





# Create a TransformerModel

model = ClassificationModel('bert', 'bert-base-cased', num_labels=2, args={'reprocess_input_data': True, 'overwrite_output_dir': True},use_cuda=False)

#model = ClassificationModel('bert', 'bert-large-cased', num_labels=2, args={'reprocess_input_data': True, 'overwrite_output_dir': True},use_cuda=False)

train_df2 = pd.DataFrame({

    'text': train_df['text'].replace(r'\n', ' ', regex=True),

    'label': train_df['target']

})



eval_df2 = pd.DataFrame({

    'text': eval_df['text'].replace(r'\n', ' ', regex=True),

    'label': eval_df['target']

})
model.train_model(train_df2)
result, model_outputs, wrong_predictions = model.eval_model(eval_df2)
print(result)

print(model_outputs)

#print(wrong_predictions)
lst = []

for arr in model_outputs:

    lst.append(np.argmax(arr))
true = eval_df2['label'].tolist()

predicted = lst
import sklearn

mat = sklearn.metrics.confusion_matrix(true , predicted)

mat
print(sklearn.metrics.classification_report(true,predicted,target_names=['fake','real']))
test_df =pd.read_csv('/kaggle/input/nlp-getting-started/test.csv' ,  encoding='ISO-8859-1')



test_df.head()
final_prediction = model.predict(list(test_df.text))
final_prediction
print('Loading in Submission File...')



submit_df = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

submit_df['target'] = final_prediction[0]



submit_df.to_csv('bert_submit.csv', index=False)
print("Finished")