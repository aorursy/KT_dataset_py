!pip install --upgrade git+https://github.com/zalandoresearch/flair.git
import pandas as pd

from pathlib import Path

from flair.data import Corpus

from flair.datasets import ClassificationCorpus

from flair.datasets import CSVClassificationCorpus

from flair.embeddings import DocumentLSTMEmbeddings, BertEmbeddings, BytePairEmbeddings

from flair.models import TextClassifier

from flair.trainers import ModelTrainer

from flair.visual.training_curves import Plotter

import string
data = pd.read_csv("../input/tweet-sentiment-extraction/train.csv", encoding='latin-1', na_filter=False)

data = data.rename(columns={"sentiment":"label", "text":"text"})

data.replace({"label":{"positive":2, "neutral":1, "negative":0}}, inplace=True)

data["text"] = data["text"].astype(str)

data["text"] = data["text"].replace('[{}]'.format(string.punctuation), '')

data["text"].apply(lambda s: s.strip())

data = data.loc[:, ["label", "text"]]

data = data.reset_index(drop=True)



Path("flair_data").mkdir(parents=True, exist_ok=True)

 

data['label'] = '__label__' + data['label'].astype(str)

data.iloc[0:int(len(data)*0.8)].to_csv('flair_data/train.csv', sep='\t', index = False, header = False)

data.iloc[int(len(data)*0.8):int(len(data)*0.9)].to_csv('flair_data/test.csv', sep='\t', index = False, header = False)

data.iloc[int(len(data)*0.9):].to_csv('flair_data/dev.csv', sep='\t', index = False, header = False)
data_folder = "flair_data"

corpus: Corpus = ClassificationCorpus(data_folder)

stats = corpus.obtain_statistics()

print(stats)
sentence_embeddings = [BytePairEmbeddings(language="en"), BertEmbeddings('bert-base-multilingual-uncased')]



document_embeddings = DocumentLSTMEmbeddings(sentence_embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256)



classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(), multi_label=False)



trainer = ModelTrainer(classifier, corpus)



trainer.train('./', embeddings_storage_mode='gpu', max_epochs=5)
%matplotlib inline



plotter = Plotter()

plotter.plot_training_curves('loss.tsv')