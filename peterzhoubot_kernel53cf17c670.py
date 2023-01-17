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
import os
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def main():
  os.system('wget https://github.com/see--/natural-question-answering/releases/download/v0.0.1/tokenizer_tf2_qa.zip')
  os.system('unzip tokenizer_tf2_qa.zip')
  tokenizer = BertTokenizer.from_pretrained('tokenizer_tf2_qa')
  model = hub.load(
      'https://github.com/see--/natural-question-answering/releases/download/v0.0.1/model.tar.gz')
  questions = [
      'How long did it take to find the answer?',
      'What\'s the answer to the great question?',
      'What\'s the name of the computer?']
  paragraph = '''<p>The computer is named Deep Thought.</p>.
                 <p>After 46 million years of training it found the answer.</p>
                 <p>However, nobody was amazed. The answer was 42.</p>'''

  for question in questions:
    question_tokens = tokenizer.tokenize(question)
    paragraph_tokens = tokenizer.tokenize(paragraph)
    tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + paragraph_tokens + ['[SEP]']
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = [0] * (1 + len(question_tokens) + 1) + [1] * (len(paragraph_tokens) + 1)

    input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0), (input_word_ids, input_mask, input_type_ids))
    outputs = model([input_word_ids, input_mask, input_type_ids])
    # using `[1:]` will enforce an answer. `outputs[:][0][0]` is the ignored '[CLS]' token logit.
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    print(f'Question: {question}')
    print(f'Answer: {answer}')


if __name__ == '__main__':
  main()
import matplotlib.pyplot as plt
import seaborn as sns

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
#sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (16,8)
#os.system('wget https://github.com/see--/natural-question-answering/releases/download/v0.0.1/tokenizer_tf2_qa.zip')
os.system('unzip tokenizer_tf2_qa.zip')
tokenizer = BertTokenizer.from_pretrained('tokenizer_tf2_qa')
model = hub.load(
  'https://github.com/see--/natural-question-answering/releases/download/v0.0.1/model.tar.gz')
questions = [
  'How long did it take to find the answer?',
  'What\'s the answer to the great question?',
  'What\'s the name of the computer?']
paragraph = '''<p>The computer is named Deep Thought.</p>.
             <p>After 46 million years of training it found the answer.</p>
             <p>However, nobody was amazed. The answer was 42.</p>'''
question=questions[0]
question_tokens = tokenizer.tokenize(question)
paragraph_tokens = tokenizer.tokenize(paragraph)
tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + paragraph_tokens + ['[SEP]']
input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
input_mask = [1] * len(input_word_ids)
input_type_ids = [0] * (1 + len(question_tokens) + 1) + [1] * (len(paragraph_tokens) + 1)

input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
    tf.convert_to_tensor(t, dtype=tf.int32), 0), (input_word_ids, input_mask, input_type_ids))
outputs = model([input_word_ids, input_mask, input_type_ids])
# using `[1:]` will enforce an answer. `outputs[:][0][0]` is the ignored '[CLS]' token logit.
short_start = tf.argmax(outputs[0][0][1:]) + 1
short_end = tf.argmax(outputs[1][0][1:]) + 1
answer_tokens = tokens[short_start: short_end + 1]
answer = tokenizer.convert_tokens_to_string(answer_tokens)
print(f'Question: {question}')
print(f'Answer: {answer}')
token_labels = []
for (i, token) in enumerate(tokens):
    token_labels.append('{:} - {:>2}'.format(token, i))
start_scores = outputs[0][0][1:].numpy()
start_scores
len(token_labels[1:49])
len(outputs[0][0][1:])
ax = sns.barplot(x=token_labels[1:49], y=start_scores, ci=None)

# Turn the xlabels vertical.
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

# Turn on the vertical grid to help align words to scores.
ax.grid(True)

plt.title('Start Word Scores')

plt.show()