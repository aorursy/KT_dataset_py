import os
import ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

pd.options.display.max_rows = 20
sns.set(style="darkgrid")
%matplotlib inline
train_sample = pd.DataFrame()
files_directory = os.listdir("../input/train_simplified")
for file in files_directory:
    train_sample = train_sample.append(pd.read_csv('../input/train_simplified/' + file, index_col='key_id', nrows=10))
# Shuffle data
train_sample = shuffle(train_sample, random_state=123)

train = pd.DataFrame()
for file in files_directory[:185]:
    train = train.append(pd.read_csv('../input/train_simplified/' + file, index_col='key_id', usecols=[1, 2, 3, 5]))
# Shuffle data
train = shuffle(train, random_state=123)
print('Train number of rows: ', train.shape[0])
print('Train number of columns: ', train_sample.shape[1])
print('Train set features: %s' % train_sample.columns.values)
print('Train number of label categories: %s' % len(files_directory))
train_sample.head()
count_gp = train.groupby(['word']).size().reset_index(name='count').sort_values('count', ascending=False)
top_10 = count_gp[:10]
bottom_10 = count_gp[count_gp.shape[0]-10:count_gp.shape[0]]
ax_t10 = sns.barplot(x="word", y="count", data=top_10, palette="coolwarm")
ax_t10.set_xticklabels(ax_t10.get_xticklabels(), rotation=40, ha="right")
plt.show()
ax_b10 = sns.barplot(x="word", y="count", data=bottom_10, palette="BrBG")
ax_b10.set_xticklabels(ax_b10.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()
count_gp
sns.countplot(x="recognized", data=train)
plt.show()
rec_gp = train.groupby(['word', 'recognized']).size().reset_index(name='count')
rec_true = rec_gp[(rec_gp['recognized'] == True)].rename(index=str, columns={"recognized": "recognized_true", "count": "count_true"})
rec_false = rec_gp[(rec_gp['recognized'] == False)].rename(index=str, columns={"recognized": "recognized_false", "count": "count_false"})
rec_gp = rec_true.set_index('word').join(rec_false.set_index('word'), on='word')
rec_gp
words = train['word'].tolist()
drawings = [ast.literal_eval(pts) for pts in train[:9]['drawing'].values]

plt.figure(figsize=(10, 10))
for i, drawing in enumerate(drawings):
    plt.subplot(330 + (i+1))
    for x,y in drawing:
        plt.plot(x, y, marker='.')
        plt.tight_layout()
        plt.title(words[i]);
        plt.axis('off')