!pip install --upgrade git+https://github.com/zalandoresearch/flair.git
from flair.data_fetcher import NLPTaskDataFetcher

from flair.embeddings import DocumentLSTMEmbeddings, BertEmbeddings, BytePairEmbeddings

from flair.models import TextClassifier

from flair.trainers import ModelTrainer

from flair.visual.training_curves import Plotter

from flair.data import Corpus

from flair.datasets import ClassificationCorpus

from pathlib import Path

import pandas as pd
data = pd.read_csv("../input/nlp-getting-started/train.csv", encoding='latin-1').sample(frac=1).drop_duplicates()

data = data[["target", "text"]].rename(columns={"target":"label", "text":"text"})

data["text"] = data["text"].str.replace("/|'|\"|:|;|@|&|\.|~|#|-|\n|\*", "", regex=True)



Path("flair_data").mkdir(parents=True, exist_ok=True)

 

data['label'] = '__label__' + data['label'].astype(str)

data.iloc[0:int(len(data)*0.8)].to_csv('flair_data/train.csv', sep='\t', index = False, header = False)

data.iloc[int(len(data)*0.8):int(len(data)*0.9)].to_csv('flair_data/test.csv', sep='\t', index = False, header = False)

data.iloc[int(len(data)*0.9):].to_csv('flair_data/dev.csv', sep='\t', index = False, header = False)
data_folder = "flair_data"

corpus: Corpus = ClassificationCorpus(data_folder)

stats = corpus.obtain_statistics()

print(stats)
word_embeddings = [BytePairEmbeddings(language="en"), BertEmbeddings('bert-base-multilingual-uncased')]



document_embeddings = DocumentLSTMEmbeddings(word_embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256)



classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(), multi_label=False)



trainer = ModelTrainer(classifier, corpus)



trainer.train('./', embeddings_storage_mode='cpu', max_epochs=200)



plotter = Plotter()

plotter.plot_training_curves('loss.tsv')