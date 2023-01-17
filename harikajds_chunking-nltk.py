import nltk

from nltk import pos_tag

from nltk import RegexpParser



#Text

txt ="This is NLP Chunking NoteBook"

text = txt.split()

print("After Split:",text)
#POS Tags

POS_tag = pos_tag(text)



print("After POS tags:",POS_tag)

#Chunking Form



patterns= """mychunk:{<NN.?>*<VBD.?>*<JJ.?>*<CC>?}"""

chunker = RegexpParser(patterns)

print("After Regex:",chunker)
output = chunker.parse(POS_tag)

print("After Chunking",output)

#output.draw()