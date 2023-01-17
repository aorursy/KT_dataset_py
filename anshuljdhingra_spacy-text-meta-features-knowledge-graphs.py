from collections import Counter 

import pandas as pd

import spacy 

nlp = spacy.load('en')
feats = ['char_count', 'word_count', 'word_count_cln',

       'stopword_count', '_NOUN', '_VERB', '_ADP', '_ADJ', '_DET', '_PROPN',

       '_INTJ', '_PUNCT', '_NUM', '_PRON', '_ADV', '_PART', '_amod', '_ROOT',

       '_punct', '_advmod', '_auxpass', '_nsubjpass', '_ccomp', '_acomp',

       '_neg', '_nsubj', '_aux', '_agent', '_det', '_pobj', '_prep', '_csubj',

       '_nummod', '_attr', '_acl', '_relcl', '_dobj', '_pcomp', '_xcomp',

       '_cc', '_conj', '_mark', '_prt', '_compound', '_dep', '_advcl',

       '_parataxis', '_poss', '_intj', '_appos', '_npadvmod', '_predet',

       '_case', '_expl', '_oprd', '_dative', '_nmod']



class AutomatedTextFE:

    def __init__(self):

        self.pos_tags = ['NOUN', 'VERB', 'ADP', 'ADJ', 'DET', 'PROPN', 'INTJ', 'PUNCT',\

                         'NUM', 'PRON', 'ADV', 'PART']

        self.dep_tags = ['amod', 'ROOT', 'punct', 'advmod', 'auxpass', 'nsubjpass',\

                         'ccomp', 'acomp', 'neg', 'nsubj', 'aux', 'agent', 'det', 'pobj',\

                         'prep', 'csubj', 'nummod', 'attr', 'acl', 'relcl', 'dobj', 'pcomp', \

                         'xcomp', 'cc', 'conj', 'mark', 'prt', 'compound', 'dep', 'advcl',\

                         'parataxis', 'poss', 'intj', 'appos', 'npadvmod', 'predet', 'case',\

                         'expl', 'oprd', 'dative', 'nmod']
def _spacy_cleaning(doc):

    toks = [t for t in doc if (t.is_stop == False)]

    toks = [t for t in toks if (t.is_punct == False)]

    words = [t.lemma_ for token in toks]

    return " ".join(words)
def _spacy_features(df):

    df["clean_text"] = df[c].apply(lambda x : _spacy_cleaning(x))

    df["char_count"] = df[textcol].apply(len)

    df["word_count"] = df[c].apply(lambda x : len([_ for _ in x]))

    df["word_count_cln"] = df["clean_text"].apply(lambda x : len(x.split()))

    df["stopword_count"] = df[c].apply(lambda x : len([_ for _ in x if _.is_stop]))

    df["pos_tags"] = df[c].apply(lambda x : dict(Counter([_.head.pos_ for _ in x])))

    df["dep_tags"] = df[c].apply(lambda x : dict(Counter([_.dep_ for _ in x])))

    return df 
class AutomatedTextFE:

    def __init__(self, df, textcol):

        self.df = df

        self.textcol = textcol

        self.c = "spacy_" + textcol

        self.df[self.c] = self.df[self.textcol].apply( lambda x : nlp(x))

        

        self.pos_tags = ['NOUN', 'VERB', 'ADP', 'ADJ', 'DET', 'PROPN', 'INTJ', 'PUNCT',\

                         'NUM', 'PRON', 'ADV', 'PART']

        self.dep_tags = ['amod', 'ROOT', 'punct', 'advmod', 'auxpass', 'nsubjpass',\

                         'ccomp', 'acomp', 'neg', 'nsubj', 'aux', 'agent', 'det', 'pobj',\

                         'prep', 'csubj', 'nummod', 'attr', 'acl', 'relcl', 'dobj', 'pcomp', \

                         'xcomp', 'cc', 'conj', 'mark', 'prt', 'compound', 'dep', 'advcl',\

                         'parataxis', 'poss', 'intj', 'appos', 'npadvmod', 'predet', 'case',\

                         'expl', 'oprd', 'dative', 'nmod']

        

    def _spacy_cleaning(self, doc):

        tokens = [token for token in doc if (token.is_stop == False)\

                  and (token.is_punct == False)]

        words = [token.lemma_ for token in tokens]

        return " ".join(words)

        

    def _spacy_features(self):

        self.df["clean_text"] = self.df[self.c].apply(lambda x : self._spacy_cleaning(x))

        self.df["char_count"] = self.df[self.textcol].apply(len)

        self.df["word_count"] = self.df[self.c].apply(lambda x : len([_ for _ in x]))

        self.df["word_count_cln"] = self.df["clean_text"].apply(lambda x : len(x.split()))

        

        self.df["stopword_count"] = self.df[self.c].apply(lambda x : 

                                                          len([_ for _ in x if _.is_stop]))

        self.df["pos_tags"] = self.df[self.c].apply(lambda x :

                                                    dict(Counter([_.head.pos_ for _ in x])))

        self.df["dep_tags"] = self.df[self.c].apply(lambda x :

                                                    dict(Counter([_.dep_ for _ in x])))

        

    def _flatten_features(self):

        for key in self.pos_tags:

            self.df["_" + key] = self.df["pos_tags"].apply(lambda x : \

                                                           x[key] if key in x else 0)

        

        for key in self.dep_tags:

            self.df["_" + key] = self.df["dep_tags"].apply(lambda x : \

                                                           x[key] if key in x else 0)

                

    def generate_features(self):

        self._spacy_features()

        self._flatten_features()

        self.df = self.df.drop([self.c, "pos_tags", "dep_tags", "clean_text"], axis=1)

        return self.df
def spacy_features(df, tc):

    fe = AutomatedTextFE(df, tc)

    return fe.generate_features()
path = "../input/ted-talks/transcripts.csv"

df = pd.read_csv(path)[:10]

textcol = "transcript"



feats_df = spacy_features(df, textcol)

feats_df[[textcol] + feats].head()
path = "../input/seinfeld-chronicles/scripts.csv"

df = pd.read_csv(path)[:10]

textcol = "Dialogue"



feats_df = spacy_features(df, textcol)

feats_df[[textcol] + feats].head()
path = "../input/news-aggregator-dataset/uci-news-aggregator.csv"

df = pd.read_csv(path)[:1500]

df["spacy_title"] = df["TITLE"].apply(lambda x : nlp(x))
pos_chain_1 = "NNP-VBZ-NNP"
df["named_entities"] = df["spacy_title"].apply(lambda x : x.ents)

for i, r in df.iterrows():

    pos_chain = "-".join([d.tag_ for d in r['spacy_title']])

    if pos_chain_1 in pos_chain:

        if len(r["named_entities"]) == 2:

            print (r["TITLE"])

            print (r["named_entities"])

            print ()

    