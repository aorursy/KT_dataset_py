import os
import ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import datetime
 

pd.options.display.max_rows = 20
sns.set(style="darkgrid")
%matplotlib inline
train_sample = pd.DataFrame()
files_directory = os.listdir("../input/train_simplified")
files_directory[:10], len(files_directory) # count labels
for file in files_directory:
    train_sample = train_sample.append(pd.read_csv('../input/train_simplified/' + file, index_col='key_id', nrows=10))
# Shuffle data
train_sample = shuffle(train_sample, random_state=123)
print(train_sample.shape)
train_sample.head(n=30)
# because of momory and speed problem, limit the numbers of each label data 10000 rows
train = pd.DataFrame()
for i, file in enumerate(files_directory):
    train = train.append(pd.read_csv('../input/train_simplified/' + file, index_col='key_id', usecols=[1, 2, 3, 5], nrows=10000))
    if i % 20 == 0:
        print(i, "[Done]", datetime.datetime.now(), file)

# Shuffle data
print("[Start] Shuffle!")
train = shuffle(train, random_state=123)
print(train.shape) # tooooooooooooooooooooooooooooooooooooooooooooooooo large, and this just have only 184 label files
train.head()
print('Train number of rows: ', train.shape[0])
print('Train number of columns: ', train_sample.shape[1])
print('Train set features: %s' % train_sample.columns.values)
print('Train number of label categories: %s' % len(files_directory))
sns.countplot(x="recognized", data=train)
plt.show()
rec_gp = train.groupby(['word', 'recognized']).size().reset_index(name='count')
rec_true = rec_gp[(rec_gp['recognized'] == True)].rename(index=str, columns={"recognized": "recognized_true", "count": "count_true"})
rec_false = rec_gp[(rec_gp['recognized'] == False)].rename(index=str, columns={"recognized": "recognized_false", "count": "count_false"})
rec_gp = rec_true.set_index('word').join(rec_false.set_index('word'), on='word')
rec_gp
words = train['word'].tolist()
drawings = [ast.literal_eval(pts) for pts in train[:9]['drawing'].values]
words[:10], drawings[-1]
len(words)
plt.figure(figsize=(10, 10))
for i, drawing in enumerate(drawings):
    plt.subplot(330 + (i+1))
    for x,y in drawing:
        plt.plot(x, y, marker='.')
        plt.tight_layout()
        plt.title(words[i]);
        plt.axis('off')
print(eval(owls_recognized['drawing'][0])) # drawing is string, so using eval function to convert list types

