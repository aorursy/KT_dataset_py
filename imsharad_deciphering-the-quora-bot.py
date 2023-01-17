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
t_file['Common'] = t_file.apply(lambda row: len(list(set(row['question1']).intersection(row['question2']))), axis=1)

t_file['Average'] = t_file.apply(lambda row: 0.5*(len(row['question1'])+len(row['question2'])), axis=1)

t_file['Percentage'] = t_file.apply(lambda row: row['Common']*100.0/row['Average'], axis=1)
cor_file = t_file[t_file.is_duplicate ==1]

cor_file2 = cor_file[['Percentage','Common']]

x = cor_file2['Common']

y = cor_file2['Percentage']

plt.scatter(x,y)



plt.xlabel("Word Length")

plt.ylabel("Percentage of Similarity")





plt.show()



cor_file = t_file[t_file.is_duplicate ==0]

cor_file2 = cor_file[['Percentage','Common']]

x = cor_file2['Common']

y = cor_file2['Percentage']

plt.scatter(x,y)



plt.xlabel("Word Length")

plt.ylabel("Percentage of Similarity")





plt.show()