text = "I think that john stone is a nice guy. the date was 17 july. there is a stone on the grass. i'm fat. are you welcome and smart in london? is this martin's dog?"
text1 = 'the date was 17 july.'
!pip install stanfordnlp
from nltk.tokenize import sent_tokenize

import re

import stanfordnlp

stanfordnlp.download('en')
def truecaser(text):    

    sentences = sent_tokenize(text, language='english')



    sentences_capitalized = [s.capitalize() for s in sentences]

    text2 = re.sub("(?=[\.,'!?:;])", "", ' '.join(sentences_capitalized))



    stf_nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos')

    doc = stf_nlp(text2)



    text3= [w.text.capitalize() if w.upos in ["PROPN","NNS"] else w.text for sent in doc.sentences for w in sent.words]

    text_truecase = re.sub("(?=[\.,'!?:;])", "", ' '.join(text3))



    return text_truecase



truecaser(text)