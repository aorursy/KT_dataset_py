import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



try:

    t_file = pd.read_csv('../input/questions.csv', encoding='ISO-8859-1')

    print('File load: Success')

except:

    print('File load: Failed')
from nltk.corpus import stopwords

stop = stopwords.words('english')

print(stop)
t_file = t_file.dropna()

t_file['question1'] = t_file['question1'].str.lower().str.split()

t_file['question2'] = t_file['question2'].str.lower().str.split()

t_file['question1'] = t_file['question1'].apply(lambda x: [item for item in x if item not in stop])

t_file['question2'] = t_file['question2'].apply(lambda x: [item for item in x if item not in stop])
t_file = t_file.apply(lambda row: len(list(set(row['question1']).intersection(row['question2']))), axis=1)
print(t_file)
t_true = t_file[t_file['is_duplicate']==0]

t_false = t_file[t_file['is_duplicate']==1]
f = 1300

t = 770

t_true['average'] = (t_true['question1'].apply(lambda x: len(x)) + t_true['question2'].apply(lambda x: len(x)))/2;

t_true['percentage']= t_true['CommonLength']*20000.0/(len(t_true['question1'])+len(t_true['question2']))

t_false['percentage']= t_false['CommonLength']*20000.0/(len(t_false['question1'])+len(t_false['question2']))
print(t_true)
fig, ax = plt.subplots()

ax.plot(t_true['percentage'][:t], label='True')

ax.plot(t_false['precentage'][:t], label='False')