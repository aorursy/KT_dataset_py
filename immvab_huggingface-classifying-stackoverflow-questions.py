!pip install -q git+https://github.com/huggingface/transformers.git
from transformers import pipeline # HuggingFace Transformers Package

import pandas as pd

from tqdm import tqdm
df = pd.read_csv('/kaggle/input/60k-stack-overflow-questions-with-quality-rate/data.csv')

df.head()
classifier = pipeline("zero-shot-classification",device = 0) 
df['Tags'] = df['Tags'].apply(lambda x: x[1:-1].split('><'))

df['Body'] = df['Body'].apply(lambda x: x[3:-4])
# Extracting the unique labels from the first 100 Rows 



labels = []

for i in range(200):

    labels.extend(df.iloc[i,]['Tags'])

labels = set(labels)

all_labels = list(labels)
from wordcloud import WordCloud

import matplotlib.pyplot as plt



cloud = ''

for i in all_labels:

    cloud += i + ' '



plt.subplots(figsize = (8,8))



wordcloud = WordCloud (

                    background_color = 'white',

                    width = 1024,

                    height = 1024

                        ).generate(cloud)

plt.imshow(wordcloud) # image show

plt.axis('off') # to off the axis of x and y

plt.savefig('Plotly-World_Cloud.png')

plt.show()
y_pred = []

y = []

for i in tqdm(range(200)):

    titles = df.iloc[i,]['Title']

    tags = df.iloc[i,]['Tags']

    op = classifier(titles, all_labels, multi_class=True)

    labels = op['labels'] 

    scores = op['scores']

    res_dict = {label : score for label,score in zip(labels, scores)}

    sorted_dict = dict(sorted(res_dict.items(), key=lambda x:x[1],reverse = True)) #sorting the dictionary of labels in descending order based on their score

    categories = []

    for i, (k,v) in enumerate(sorted_dict.items()):

        if(i > 3): #storing only the best 4 predictions

            break

        else:

            categories.append(k)

    y.append(tags)

    y_pred.append(categories)
out = pd.DataFrame(list(zip(y, y_pred)), columns =['Labels', 'Predicted_Labels']) 

out.to_csv('output.csv')

out.head(10)
cat_idx = {cat : i for i,cat in enumerate(all_labels)}  # Map of category and it's index to encode the o/p for evaluation
y_trueEncoded = []

y_predEncoded = []

for y_true, y_pred in zip(y, y_pred):

    encTrue = [0] * len(all_labels)

    for cat in y_true:

        idx = cat_idx[cat]

        encTrue[idx] = 1

    y_trueEncoded.append(encTrue)

    encPred = [0] * len(all_labels)

    for cat in y_pred:

        idx = cat_idx[cat]

        encPred[idx] = 1

    y_predEncoded.append(encPred)
from sklearn.metrics import hamming_loss

print('Hamming Loss =', hamming_loss(y_trueEncoded,y_predEncoded))
loss = 0

for x, y in zip(y_trueEncoded,y_predEncoded):

    temp = 0

    for i in range(len(x)):

        if x[i] == y[i]:

            temp += 1

    temp /= len(x)

    loss += temp

loss /= len(y_trueEncoded)

print('Hamming Loss =', 1 - loss)