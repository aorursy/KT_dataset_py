import pandas as pd
# Visualization

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from IPython.display import Image
import pprint # pretty printing
# Natural language processing

import spacy

from gensim.utils import simple_preprocess
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel

import pyLDAvis
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
import warnings # remove all the deprecation and future warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning) 
# Download the model with 'python -m spacy download en_core_web_lg --user'
# Note: this is a 800MB model
#       'en_core_web_sm', 29MB can be used as alternative
#       see https://spacy.io/models/en#section-en_core_web_lg
nlp = spacy.load('en_core_web_lg')
df = pd.read_csv('../input/ETHPrize Developer Interviews.csv')

# Anonymize answers (interviewees are OK to have their names public)
# df.drop(columns=['Name'], inplace = True)
# View a snippet of the data
df.head(3)
# View the counts of answers and non-empty answers
df.describe()
# What does a full answer look like:
df['What are the tools/libraries/frameworks you use?'][0]
df.columns = [
    'Name',              # Name
    'tooling',           # What are the tools/libraries/frameworks you use?
    'frustrations',      # What are your biggest frustrations?
    'testing',           # How do you handle testing?
    'smart_contract',    # How do you handle smart contract verif & security?
    'bounties',          # Other bounties
    'who_what',          # Who are you and what are you working on?
    'domain_questions',  # Other domain specific questions?
    'missing_tools',     # What tools don’t exist at the moment?
    'easier_expected',   # Was anything easier than expected?
    'excited_about',     # What are you most excited about in the short term?
    'hardest_part',      # What was the hardest part to develop with Ethereum?
    'best_resources',    # What are the best educational resources?
    'questions_to_ask',  # Are there any other questions we should be asking?
    'people_talk_to'     # Who do you think we should talk to?
]

df.fillna('', inplace = True)
df.head(3)
for i in range(10):
    print(f"""
####################################
doc {i}, number of characters: {len(df['tooling'][i])}
""")
    if df['tooling'][i] != '':
        doc = nlp(df['tooling'][i]) # nlp = spacy.load('en_core_web_lg')
        print(doc)
del doc
for i in [0, 5, 8]:
    print(f"""
####################################
doc {i}, number of characters: {len(df['tooling'][i])}

document | {"Label".rjust(20)} | {"Entity type".rjust(15)}""")
    doc = nlp(df['tooling'][i])
    for entity in doc.ents:
        print(f'{i:>8} | {entity.text.rjust(20)} | {entity.label_.rjust(15)}')
def clean_punct(txt):
    x = txt.replace("\n", ". ")
    x = x.replace(" -- ", ": ")
    x = x.replace(" - ", ": ")
    return x

def reCase(txt):
    ## recasing common words so that spaCy picks them up as entities
    ## an ethereum specific NLP model shouldn't need that
    ## Also this is inefficient as we could replace everything in one pass
    ## but we'll worry about that for 10k+ interviews.
    x = txt.replace("solidity", "Solidity")
    x = x.replace("truffle", "Truffle")
    x = x.replace(" eth", " Eth") # space to avoid recasing geth into gEth
    x = x.replace(" geth", " Geth") # avoid together -> toGether ¯_(ツ)_/¯
    x = x.replace("jQuery", "JQuery")
    x = x.replace(" react", " React")
    x = x.replace(" redux", " Redux")
    x = x.replace("testRPC", "TestRPC")
    x = x.replace("keythereum", "Keythereum")
    # ...
    return x
for i in [0, 5, 8]:
    print(f"""
####################################
doc {i}, number of characters: {len(df['tooling'][i])}

document | {"Label".rjust(20)} | {"Entity type".rjust(15)}""")
    doc = nlp(reCase(clean_punct(df['tooling'][i])))
    for entity in doc.ents:
        print(f'{i:>8} | {entity.text.rjust(20)} | {entity.label_.rjust(15)}')
del doc
# Remove noise 
ignore_words = {
    'Ethereum',
    'UI',
    'ETH', 'Eth',
    'IDE',
    'ABI'
}

def tooling_extraction(txt):
    if txt == '':
        return ''
    doc = nlp(reCase(clean_punct(txt)))
    tools = []
    for named_entity in doc.ents:
        if named_entity.label_ in {'PERSON', 'ORG', 'PRODUCT'} and named_entity.text not in ignore_words:
            txt = named_entity.text.replace(' ', '_')
            tools.append(txt)
    return ', '.join(tools)
df['tooling_extracted'] = df['tooling'].apply(tooling_extraction)
# Reminder - a lot is missed due to the very niche domain
# while the NLP model was trained on general web publications.
df['tooling_extracted']
# Concatenating all tools
tools = ''
for idx, row in df['tooling_extracted'].iteritems():
    tools += ', ' + row 

# Setting up the figure
# Don't ask me what title and suptitle (super-title) are supposed to do ...
plt.figure(figsize=(12,6))
wordcloud = WordCloud(background_color='white', width=500, height=300,
                      max_font_size=50, max_words=80).generate(tools)
plt.imshow(wordcloud)
plt.title("""
ETHPrize devs - tools, library frameworks

""", fontsize=20)
plt.suptitle("""This is a basic approach, manual processing is recommended.
At scale tagging 10~20% of the data manually
before automated extraction on the rest would probably be much better
""", fontsize=14)

plt.axis("off")
plt.show()

# Cleanup
del tools
for i in range(10):
    print(f"""
####################################
doc {i}, number of characters: {len(df['frustrations'][i])}
""")
    if df['frustrations'][i] != '':
        doc = nlp(df['frustrations'][i]) # nlp = spacy.load('en_core_web_lg')
        print(doc)
del doc
doc = nlp(df['frustrations'][0][-299:])
for token in doc:
    print(token.text)
del doc
# Lets have a look at the included stopwords

print(nlp.Defaults.stop_words)
def topic_modelling_preprocess(txt):    
    # Remove stopwords
    stop_words = nlp.Defaults.stop_words
    stop_words.add('ethereum')
    stop_words.add('frustrations')
    
    cleaned = [word for word in simple_preprocess(txt, deacc = True) if word not in stop_words]
    if cleaned == []:
        return []
    
    # Lemmatization - TODO improve
    doc = nlp(" ".join(cleaned))    
    result = [token.lemma_ for token in doc if token.pos_ in {'NOUN', 'ADJ', 'VERB', 'ADV'}]
    return result
# Let's try it on doc 5 which is short (394 characters)

print('#### Before\n')
print(df['frustrations'][5])

print('#### After\n')
topic_modelling_preprocess(df['frustrations'][5])
# Let's start with a quick and dirty model

frustrations = df['frustrations'].apply(topic_modelling_preprocess)
frustrations
# Create Dictionary
id2word = corpora.Dictionary(frustrations)

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in frustrations]

# Modelization, nuber of topics need to be tuned
lda_model = LdaModel(corpus=corpus,
                     id2word=id2word,
                     num_topics=10,    # <--- tune this
                     random_state=1337,
                     update_every=1,
                     chunksize=100,
                     passes=10,
                     alpha='auto',
                     per_word_topics=True)
pprint.pprint(lda_model.print_topics())
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis
del id2word
del corpus
del frustrations
del vis
def gen_stop_words(extra_stop_words):
    ## Expect a list of stop words in lowercase
    result = nlp.Defaults.stop_words
    for word in extra_stop_words:
        result.add(word)
    return result
def model_topics(txts, num_topics, stop_words, use_ngrams = False, n_grams_min_count = 5, n_grams_score_threshold = 1):
    # Note: here we process the whole dataframe series
    
    # Transform the serie into a list of list of words,
    # Remove stopwords at the same time
    cleaned = []
    for idx, txt in txts.iteritems():
        # Remove stopwords
        cleaned += [[word for word in simple_preprocess(txt, deacc = True) if word not in stop_words]]
    
    if use_ngrams:
        # Build bigrams and trigrams
        bigrams = Phraser(Phrases(cleaned, min_count=n_grams_min_count, threshold=n_grams_score_threshold))
        trigrams = Phraser(Phrases(bigrams[cleaned], threshold=n_grams_score_threshold))
        
        # Now create the bag of words with the new trigrams
        cleaned = [trigrams[bigrams[txt]] for txt in cleaned]
    
    # Lemmatization - TODO improve
    lemmatized = []
    for txt in cleaned:
        if txt == []:
            lemmatized += []
        else:
            doc = nlp(" ".join(txt))    
            lemmatized += [[token.lemma_ for token in doc if token.pos_ in {'NOUN', 'ADJ', 'VERB', 'ADV'}]]
            
    print("Snippet of keywords for topic modelling for the first 3 answers")
    print(lemmatized[0:3])
        
    # Create Dictionary
    id2word = corpora.Dictionary(lemmatized)

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in lemmatized]
    
    # Modelling
    lda_model = LdaModel(corpus=corpus,
                         id2word=id2word,
                         num_topics=num_topics,
                         random_state=1337,
                         update_every=1,
                         chunksize=100,
                         passes=10,
                         alpha='auto',
                         per_word_topics=True)
    
    ## Model performance
    print("\nModel performance\n")
    
    ## Perplexity
    print(f"""Perplexity: {lda_model.log_perplexity(corpus)}. Lower is better.
    See https://en.wikipedia.org/wiki/Perplexity.
    The best number of topics minimize perplexity.
    """)
    
    ## Coherence
    coherence = CoherenceModel(
        model=lda_model,
        texts=lemmatized,
        dictionary=id2word,
        coherence='c_v'
    )
    ## Corpus coherence
    print(f'Whole model coherence: {coherence.get_coherence()}.')
    
    ## By topic coherence
    topic_coherences = coherence.get_coherence_per_topic()
    print(f"""
By topic coherence. Higher is better.
    Measure how "well related" are the top words within the same topic.
    """)
    
    print(f'topic_id | {"top 3 keywords".rjust(45)} | topic coherence')
    for topic_id in range(num_topics):
        words_proba = lda_model.show_topic(topic_id, topn=3)
        words = [words for words,proba in words_proba]
        print(f'{topic_id:>8} | {str(words).rjust(45)} | {topic_coherences[topic_id]:>8.4f}')
    
    return lda_model, corpus, id2word
stop_words = gen_stop_words(['ethereum', 'frustrations'])

lda_model, corpus, id2word = model_topics(
    df['frustrations'], 10, stop_words,
    use_ngrams = True, n_grams_min_count = 1, n_grams_score_threshold = 1 # Need more data to have less permissive thresholds
)
pprint.pprint(lda_model.print_topics())
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis # Interactive vizualization, doesn't work on Github
stop_words = gen_stop_words(['ethereum', 'tool', 'tooling', 'tools'])

lda_model, corpus, id2word = model_topics(
    df['tooling'], 7, stop_words,
    use_ngrams = True, n_grams_min_count = 1, n_grams_score_threshold = 1
    # Need more data to have less permissive thresholds
)
pprint.pprint(lda_model.print_topics())
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis # Interactive vizualization, doesn't work on Github
stop_words = gen_stop_words(['ethereum'])

lda_model, corpus, id2word = model_topics(
    df['who_what'], 7, stop_words,
    use_ngrams = True, n_grams_min_count = 1, n_grams_score_threshold = 1
    # Need more data to have less permissive thresholds
)
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis # Interactive vizualization, doesn't work on Github
stop_words = gen_stop_words(['ethereum', 'exciting', 'excited'])

lda_model, corpus, id2word = model_topics(
    df['excited_about'], 7, stop_words,
    use_ngrams = True, n_grams_min_count = 1, n_grams_score_threshold = 1
    # Need more data to have less permissive thresholds
)
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis # Interactive vizualization, doesn't work on Github