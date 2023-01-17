import nltk
nltk.download('reuters')
from nltk.corpus import reuters
reuters_corpora = [reuters.raw(fid) for fid in reuters.fileids()]
# !python -m spacy download en_core_web_sm
import spacy
from spacy.symbols import DET, X, NUM, PRON
from tqdm import tqdm
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore

nlp = spacy.load('en_core_web_sm')
processed_corpora = [
    [token.lemma_ for token in nlp(doc_text) \
        if not(token.is_punct or token.is_stop or token.is_space or token.pos in [DET, NUM, PRON])]
    for doc_text in tqdm(reuters_corpora)]
# Create Dictionary
gensim_dict = corpora.Dictionary(processed_corpora)

# gensim_dict.token2id = reuters_vocab
processed_corpora = [gensim_dict.doc2bow(text) for text in processed_corpora]
lda_model = LdaMulticore(processed_corpora,
                        id2word=gensim_dict,
                        num_topics=10,
                        workers= 2)
lda_model
lda_model.print_topics()
from gensim.test.utils import datapath

model_file = datapath("gensim_model")
lda_model.save(model_file)

# You can load it using:
# from gensim.models.ldamulticore import LdaModel
# lda_model = LdaModel.load(model_file)
import pyLDAvis
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, processed_corpora, gensim_dict)
vis
from spacy.tokens import Doc
topics = lambda doc: lda_model[gensim_dict.doc2bow([token.lemma_ for token in doc])][0]
Doc.set_extension('topics', getter=topics)

doc = nlp(u'The decisive factor now is the behavior of the U.S. president, who basically told the crown prince, we are giving you free rein as long as you buy enough weapons and other things from us')
print(doc._.topics)
