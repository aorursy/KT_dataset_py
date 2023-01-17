import pandas as pd

import re
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
irony_data = pd.read_csv('/kaggle/input/ironic-corpus/irony-labeled.csv')

irony_data.head()
irony_data.shape
irony_data.label.value_counts()
unironic = irony_data['comment_text'][irony_data.label == -1].values

ironic = irony_data['comment_text'][irony_data.label == 1].values
unironic[:5]
ironic[:5]
def count_words(lines,linetype):

    total_words = 0

    for line in lines:

        total_words += len(re.findall(r'\w+', line))

    print(f'Number of {linetype} comments: {len(lines)}, Total words: {total_words}, Words per comment: {total_words / len(lines)}')
count_words(unironic, "Unironic")

count_words(ironic, "Ironic")