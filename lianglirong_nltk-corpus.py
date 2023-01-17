from nltk.corpus import gutenberg
from nltk.corpus import webtext
from nltk.corpus import nps_chat
from nltk.corpus import brown
from nltk.corpus import reuters
from nltk.corpus import inaugural
import nltk
print(gutenberg.fileids())
for fileid in webtext.fileids():
    print(fileid,webtext.raw(fileid)[:65])
for fileid in nps_chat.fileids():
    print(fileid,nps_chat.raw(fileid)[:60])
for fileid in nps_chat.fileids():
    print(fileid,nps_chat.posts(fileid))
brown.categories()
brown.words(categories="news")
print(brown.fileids())
print(reuters.fileids())
print(inaugural.fileids())
