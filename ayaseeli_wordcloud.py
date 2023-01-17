import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)

test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

train_qs
import matplotlib.pyplot as plt

from wordcloud import WordCloud

cloud = WordCloud(width=1440, height=1080).generate(" ".join(train_qs.astype(str)))

plt.figure(figsize=(15, 10))

plt.imshow(cloud)

plt.axis('off')