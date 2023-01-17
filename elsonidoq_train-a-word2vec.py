from pathlib import Path



DATA_PATH = Path('/kaggle/input/CORD-19-research-challenge/2020-03-13/')

JUST_SOME = True # helpful for testing the code with small data
from tqdm.auto import tqdm



import json



def iter_texts():

    """

    Iterate over all directories, all file names, and yield all elements on body_text and abstract

    """

    dirs = 'comm_use_subset noncomm_use_subset pmc_custom_license biorxiv_medrxiv'.split()

    for dir in dirs:

        fnames = (DATA_PATH / dir / dir).glob('*')

        for fname in fnames:

            with fname.open() as f:

                content = json.load(f)

                

            for key in 'abstract body_text'.split():

                for row in content[key]:

                    yield row['text']
import spacy



# make sure to run python3 -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_sm")



def iter_sents(just_some=False):

    """

    Use spacy to tokenize what's yielded by iter_sents

    """

    for i, text in enumerate(iter_texts()):

        if just_some and i == 1000: break

        doc = nlp(text)

        for sent in doc.sents:

            yield [t.text.lower() for t in sent if not t.is_stop and t.is_alpha and len(t.text) > 1]
import json



# dump on a jsonlines file so we do the tokenization just once

with open('all_sentences.jl', 'w') as f:

    for i, sent in enumerate(tqdm(iter_sents(just_some=JUST_SOME))):

        if i > 0: f.write('\n')

        f.write(json.dumps(sent))
class CachedSentenceIterator:

    """

    An iterator that is compatible with gensim (you need to be able to iterate this more than once for the epochs to work)

    """

    def __init__(self, just_some=False, fname='all_sentences.jl'): 

        self.just_some = just_some

        self.fname = fname

    

    def __iter__(self):

        with open(self.fname) as f:

            for line in f:

                yield json.loads(line)

        
from gensim.models import Word2Vec



si = CachedSentenceIterator(just_some=JUST_SOME)



model = Word2Vec()

model.build_vocab(sentences=tqdm(si))

model.train(tqdm(si), total_examples=model.corpus_count, epochs=3)
model.save('covid.w2v')
model.wv.most_similar('virus')
model.wv.most_similar('coronavirus')