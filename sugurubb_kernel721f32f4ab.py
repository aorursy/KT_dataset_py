import pandas as pd

df = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json", lines=True)

df.head()
file2 = open('../input/bert-embeddingvector/output.txt', "r")

temp = file2.read()

file2.close()
import ast 



temp2 = temp.split("\n")

ls1 = []

for t in range(0,100):

    temp3 = ast.literal_eval(temp2[t])

    ls1.append(temp3)
artist = ls1[20]['features'][3]['layers'][0]['values']

print(df['headline'][20])
authors = ls1[63]['features'][-2]['layers'][0]['values'] 

circus =  ls1[23]['features'][-2]['layers'][0]['values']

teachers = ls1[80]['features'][3]['layers'][0]['values']

holiday = ls1[44]['features'][-3]['layers'][0]['values']
d = {'authors':authors, 'artist':artist, 'circus':circus,'teachers':teachers,'holiday':holiday}

x = pd.DataFrame(data=d)

x = x.transpose()
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
adjustDf = pd.concat([principalDf,pd.DataFrame(['authors','artist','circus','teachers','holiday'])],axis=1)

adjustDf.columns = ['x', 'y', 'group']



print(adjustDf)
import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import seaborn as sns
# basic plot

p1=sns.regplot(data=adjustDf, x="x", y="y", fit_reg=False, marker="o", color="skyblue", scatter_kws={'s':400})



# add annotations one by one with a loop

for line in range(0,adjustDf.shape[0]):

     p1.text(adjustDf.x[line]+0.2, adjustDf.y[line], adjustDf.group[line], horizontalalignment='left', size='medium', color='black', weight='semibold')

plt.title("PCA analysis on word embedding vector from BERT")

plt.show()