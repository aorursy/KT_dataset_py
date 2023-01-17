import os
import pandas as pd
import codecs
import regex
#checking on which bibles are available
pd.read_csv("../input/bible_version_key.csv")
#download the data
!wget -qO- --show-progress https://wit3.fbk.eu/archive/2016-01//texts/de/en/de-en.tgz | tar xz; mv de-en corpora
#checking out what the training data looks like.
open("corpora/train.tags.de-en.de").read()[:1000]
def _refine(line):
    line = regex.sub("<[^>]+>", "", line)
    line = regex.sub("[^\s\p{Latin}']", "", line) 
    return line.strip()
de_sents = [_refine(line) for line in codecs.open('./corpora/IWSLT16.TED.tst2014.de-en.de.xml', 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]
de_sents
#loading in the text from both versions of the bible
new_bib = pd.read_csv("../input/t_web.csv").t
old_bib = pd.read_csv("../input/t_kjv.csv").t
# a regex to remove the curly braces and the notes inside them. 
def remove_notes(notes):
    return regex.sub("[\{\[].*?[\}\]]", "", notes)
new_bib = new_bib.apply(remove_notes)
old_bib = old_bib.apply(remove_notes)
