import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

f = open("/kaggle/input/gutenberg/gutenberg/austen-persuasion.txt", "r")
raw = f.read()
from nltk import word_tokenize
tokens = word_tokenize(raw.lower())
import nltk
text = nltk.Text(tokens)
text[1024:1062]
raw.find("other children")
text.findall(r"<other> <children>")
len(text)
text.count("mary")
text.index("mary")
dist1 = nltk.FreqDist(text)
dist1.most_common(50)
import re
[m.start() for m in re.finditer('children', raw)]
list(nltk.bigrams(text))
fdist = nltk.FreqDist(nltk.bigrams(text))
fdist.most_common(50)
text.findall('<sir> <walter>')
import re
[m.start() for m in re.finditer('sir walter', raw.lower())]
list(nltk.trigrams(text))
nltk.FreqDist(nltk.FreqDist(nltk.trigrams(text))).most_common(50)
import re
[m.start() for m in re.finditer('she could not', raw)]
nltk.FreqDist(nltk.FreqDist(nltk.trigrams(text)))[('could', 'not', 'be')]
len(list(nltk.trigrams(text)))
len(set(list(nltk.trigrams(text))))
len(list(nltk.bigrams(text)))
len(set(list(nltk.bigrams(text))))
len(tokens)
len(set(tokens))