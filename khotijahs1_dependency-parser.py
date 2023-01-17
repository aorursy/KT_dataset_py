import spacy

sp = spacy.load('en_core_web_sm')
from spacy import displacy



sen = sp(u"I love machine learning and deep learning")

displacy.render(sen, style='dep', jupyter=True, options={'distance': 85})
for word in sen:

    print(f'{word.text:{12}} {word.pos_:{10}} {word.tag_:{8}} {spacy.explain(word.tag_)}')
sen1 = sp(u'Can you search it on kaggle ?')

word = sen1[5]



print(f'{word.text:{12}} {word.pos_:{10}} {word.tag_:{8}} {spacy.explain(word.tag_)}')
from spacy import displacy



sen1 = sp(u"Can you search it on kaggle ?")

displacy.render(sen1, style='dep', jupyter=True, options={'distance': 85})
for word in sen1:

    print(f'{word.text:{12}} {word.pos_:{10}} {word.tag_:{8}} {spacy.explain(word.tag_)}')