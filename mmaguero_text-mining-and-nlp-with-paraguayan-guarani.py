!wget --no-verbose http://homes.soic.indiana.edu/gasser/L3/ParaMorfo-1.1.tar.gz
!file /kaggle/working/ParaMorfo-1.1.tar.gz
!tar xzf  /kaggle/working/ParaMorfo-1.1.tar.gz  -C .  
!cat          ParaMorfo-1.1/README.txt      
!sed -i 's/for pos in range(1, len(word))/for pos in range(1, len(word)):/g' /kaggle/working/ParaMorfo-1.1/l3/morpho/strip.py
ls -ltr /kaggle/working/ParaMorfo-1.1
import os

os.chdir('/kaggle/working/ParaMorfo-1.1')

!python setup.py install
import l3 # ParaMorfo
l3.anal('gn', "nanemandu'áipa")
analysis = l3.anal('gn', "nanemandu'áipa", raw=True)

print(analysis)
stem = analysis[0][0]

print(stem)

features = analysis[0][1]

for k,v in features.items():

    print(k,v)
l3.gen('gn', 'guata')
l3.gen('gn', 'mba\'apo', '[sj=[+1,+p],+pos]')
l3.anal('gn', "romba'apo")
text = l3.gen('gn', 'mba\'apo', '[sj=[+1,+p],+neg]')
l3.anal('gn', "noromba'apói")
import spacy

!python -m spacy download es_core_news_md

!python -m spacy link es_core_news_md es_md

nlp = spacy.load('es_md', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed
def get_stem(text,  mixed=False):

    stem = []

    doc = nlp(text)

    for token in doc:

        if len(l3.anal('gn', token.text, raw=True)) > 0:

            try:

                token_ = l3.anal('gn', token.text, raw=True)[0][0] # use paramorfo word root

            except:

                token_ = token.text

        elif mixed:

            token_ = token.lemma_ # use spacy lemmatizer

        token = token_ if token.is_punct else " "+token_ # for delete extra space: punctuaction

        stem.append(token)

    return "".join(stem).replace("<","").strip()
# This is a sample

sentence = "Kóva ha'e peteĩ techaukaharã" # -> https://gn.wikipedia.org/wiki/Ape

print(get_stem(sentence))
# This is a sample, mixing languages

sentence = "Kóva ha'e peteĩ techaukaharã, mezclando idiomas" # -> https://gn.wikipedia.org/wiki/Ape

print(get_stem(sentence, True))
def get_tag(word):

    tag = 'u'

    if len(l3.anal('gn', word, raw=True)) > 0:

        try:

            tag = l3.anal('gn', word, raw=True)[0][1]['pos'] # use paramorfo for pos 

        except:

            tag = 'x' # undefined or missing 

    return tag
# This is a sample, although bad

sentence = "Kóva ha'e peteĩ techaukaharã, vai jepe" # -> https://gn.wikipedia.org/wiki/Ape

doc = nlp(sentence)

for token in doc:

    print(token,get_tag(token.text))   
for token in doc:

    if get_tag(token.text).lower() in ['n','noun','v','verb','adj','adv','propn']:

        print(token)
sentence = "Kóva ha'e peteĩ techaukaharã, vai jepe"

print(sentence)

doc = nlp(sentence)

for token in doc:

    if get_tag(token.text).lower() in ['n','noun','v','verb','adj','adv','propn']:

        print(get_stem(token.text))
# This is a sample, even if it's bad and not the best.

sentence = "Kóva ha'e peteĩ techaukaharã, aunque esté feo y no sea el mejor."

content_tags_es = ['adj','adv','noun','verb','propn']

content_tags_gn = ['n','v','adj','adv'] # n: noun, v: verb

exclude = ['conj'] # put here tags of words such: y, meaning: gn=water, es=conjunction

print(sentence)

doc = nlp(sentence)

for token in doc:

    tag_gn = get_tag(token.text)

    tag_es = token.pos_.lower()

    if ((tag_gn in content_tags_gn) or (tag_es in content_tags_es)): #and (tag_gn not in exclude and tag_es not in exclude): #uncomment this if you need hidden some conflicts such "y" 

        print("---",tag_gn,tag_es,get_stem(token.text, True))