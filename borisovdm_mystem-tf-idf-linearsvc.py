! pip install pymystem3
import json

import pandas as pd

import re



from nltk.corpus import stopwords

from pymystem3 import Mystem



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC



import warnings

warnings.filterwarnings('ignore')
mystem = Mystem() 

russian_stopwords = stopwords.words('russian')



data_path = '/kaggle/input/made2019ml-cluster2/'
article_doc_id = []

article_text = []



with open(data_path + 'cosmo_content_storage_final_cut.jsonl', 'r') as f:

    for line in f:

        article = json.loads(line)

        article_doc_id.append(article['doc_id'])

        article_text.append(article.get('description', '') + ' ' + article.get('title', '') + ' ' + ''.join(re.findall(r'[a-zа-я]+', article['url'].split('/')[2].lower())))

        

data = pd.DataFrame({'doc_id': article_doc_id, 'text': article_text})
with open(data_path + 'cluster_final_cut_train.json', 'r') as f:

    labeled_data = json.load(f)

    

labeled_data = {int(k): v for k, v in labeled_data.items()}
data['text'] = data['text'].apply(lambda word: ' '.join(re.findall(r'[a-zа-я]+', word.lower().replace('ё', 'е'))))

data['text'] = data['text'].apply(

    lambda b: ' '.join([x for x in mystem.lemmatize(b) if len(x) > 1 and x not in russian_stopwords])

)



data['label'] = data['doc_id'].map(labeled_data)
df_train = data[data.label.notnull()]

df_test = data[data.label.isnull()]



df_train['label'] = df_train['label'].astype('int')

df_test.drop(columns=['label'], inplace=True)



df_train = df_train.dropna().reset_index(drop=True)

df_test.reset_index(drop=True, inplace=True)
vectorizer = TfidfVectorizer()

vectorizer.fit(df_train['text'])



clf = LinearSVC()

clf.fit(vectorizer.transform(df_train['text']), df_train['label'])
preds = clf.predict(vectorizer.transform(df_test['text'].fillna('')))

df_test['cat'] = preds

df_test[['doc_id', 'cat']].to_csv('submission.csv', index=False)