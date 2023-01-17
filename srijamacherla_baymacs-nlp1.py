import pandas as pd

import os
data=pd.read_csv(os.path.join('/kaggle/input/tweet-sentiment-extraction', 'test.csv'))
data.head()
data=data.fillna(' ')
data['tokens'] = data['text'].apply(lambda x: x.split())

data
from nltk.corpus import stopwords

stop = stopwords.words('english')
data['tokens'] = data['tokens'].apply(lambda x: [i for i in x if i not in stop])
import nltk

data['pos_tags']= data['tokens'].apply(lambda x: nltk.tag.pos_tag([i.lower() for i in x]))

data.head()
data['cleaned_tags'] = data['pos_tags'].apply(lambda x: [word for word,tag in x if tag != 'NNP' and tag != 'NNPS'])
data['selected_text'] = data['cleaned_tags'].apply(lambda x: ' '.join(x))
data[['textID','selected_text']].to_csv('/kaggle/working/submission.csv',index=False)