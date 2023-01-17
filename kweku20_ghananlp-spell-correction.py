import re
import math,random
import string
from collections import Counter
with open('../input/gen-dataset/jw300.en-tw.tw',encoding="utf8") as file:
    TEXT = file.read()
len(TEXT)
def tokens(text):
    "List all the word tokens (consecutive letters) in a text. Normalize to lowercase."
    return re.findall('[a-zA-ZƆɔɛƐ]+', text.lower()) 
tokens('Ma wo dabi nyɛ dabi!123')
WORDS = tokens(TEXT)
len(WORDS)
print(WORDS[:20])
def sample(bag, n=10):
    "Sample a random n-word sentence from the model described by the bag of words."
    return ' '.join(random.choice(bag) for _ in range(n))
sample(WORDS)
Counter(tokens('Ma wo dabi nyɛ dabi? Ma wo dabi nyɛ dabi!'))
COUNTS = Counter(WORDS)

print(COUNTS.most_common(10))
for w in tokens('Ma wo dabi nyɛ dabi'):
    print(COUNTS[w], w)
def correct(word):
    "Find the best spelling correction for this word."
    # Prefer edit distance 0, then 1, then 2; otherwise default to word itself.
    candidates = (known(edits0(word)) or 
                  known(edits1(word)) or 
                  known(edits2(word)) or 
                  [word])
    return max(candidates, key=COUNTS.get)
def known(words):
    "Return the subset of words that are actually in the dictionary."
    return {w for w in words if w in COUNTS}

def edits0(word): 
    "Return all strings that are zero edits away from word (i.e., just word itself)."
    return {word}

def edits2(word):
    "Return all strings that are two edits away from this word."
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}
def edits1(word):
    "Return all strings that are one edit away from this word."
    pairs      = splits(word)
    deletes    = [a+b[1:]           for (a, b) in pairs if b]
    transposes = [a+b[1]+b[0]+b[2:] for (a, b) in pairs if len(b) > 1]
    replaces   = [a+c+b[1:]         for (a, b) in pairs for c in alphabet if b]
    inserts    = [a+c+b             for (a, b) in pairs for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def splits(word):
    "Return a list of all possible (first, rest) pairs that comprise word."
    return [(word[:i], word[i:]) 
            for i in range(len(word)+1)]

alphabet = 'abcdeɛfghijklmnoɔpqrstuvwxyz'
splits('Gyinapɛn')
print(edits0('Gyinapɛn'))
print(edits1('Gyinapɛn'))
print(known(edits1('Gyinapɛn')))
print(len(edits2('Gyinapɛn')))
print(map(correct, tokens('wɔagye asubɔ biara nni asafo nhyiam bi ase.')))
def correct_text(text):
    "Correct all the words within a text, returning the corrected text."
    return re.sub('[a-zA-ZƆɔɛƐ]+', correct_match, text)

def correct_match(match):
    "Spell-correct word in match, and preserve proper upper/lower/title case."
    word = match.group()
    return case_of(word)(correct(word.lower()))

def case_of(text):
    "Return the case-function appropriate for text: upper, lower, title, or just str."
    return (str.upper if text.isupper() else
            str.lower if text.islower() else
            str.title if text.istitle() else
            str)
map(case_of, ['PƐPƐPƐ', 'kosi', 'Francefo', 'GyinapɛnSɛnea'])
# Correct => Gyinapɛn
correct_text('Gyinapɛ')
