import pandas as pd



'''0 - negativo, 1 - positivo '''

df = pd.read_csv('/kaggle/input/labeledTrainData.tsv', delimiter='\t', encoding='utf-8')

df.head(5)
from nltk.corpus import stopwords

stop = stopwords.words('english')



df['review_no_stopwords'] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

df.head(5)
from sklearn.feature_extraction.text import CountVectorizer



vetorizador = CountVectorizer(binary = 'true')

vetorizador.fit(df['review_no_stopwords'])

from _pickle import dump

saida = open('vetorizador.pkl', 'wb')

dump(vetorizador, saida, -1)

saida.close()
