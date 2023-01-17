list_words=['viruz','Microsoft','handuclean','feeeevvvveerrrr','facemaskk','sanitized/dry','First']
corrected=[]
#Libraries which we will be using for our task
import spacy
import nltk
#!pip3 install wordninja
import wordninja
#!pip3 install pyspellchecker
from spellchecker import SpellChecker
spell = SpellChecker()
import spacy 
#python3 -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")
from collections import Counter
import re
print('List of remaining words \n',*list_words)
print('List of correct words \n',*corrected)
def NER(list_words,corrected):
    for i in (list_words):

        doc = nlp(i) 
        
        for ent in doc.ents: 
            print(ent.text, ent.label_) 
            corrected.append(ent.text)
            list_words.remove(ent.text)
            
    return [list_words,corrected]
list_words,corrected=NER(list_words,corrected)
print('List of remaining words \n',*list_words)
print('List of correct words \n',*corrected)
def preprocess_text(list_words,corrected):
  index_array=[]

  for i in ((list_words)):
    if(len(list(spell.unknown([i])))!=0):   
      index_array.append([i,spell.correction(i)])
    else:
        corrected.append(i)
        list_words.remove(i)
  
  for i in ((index_array)):
    if(len(list(spell.unknown([i[1]])))==0):
      corrected.append(i[1])
      list_words.remove(i[0])
        
  print(*corrected)
  return [list_words,corrected]
list_words,corrected=preprocess_text(list_words,corrected)
print('List of remaining words \n',*list_words)
print('List of correct words \n',*corrected)
def recurring(list_words):
  for q in range(len(list_words)):
      chars=list(list_words[q])
      col_count = Counter(chars)
      pairs=list(col_count.items())
        
      for i in range(0,len(pairs)):
        if(pairs[i][1]>2):
          pattern=pairs[i][0]+"+"
          text_input=re.sub(pattern,pairs[i][0],text_input)
        else:
            text_input=list_words[q]
    
      list_words[q]=text_input
    
  return list_words
list_words=recurring(list_words)
list_words,corrected=preprocess_text(list_words,corrected)
print('List of remaining words \n',*list_words)
print('List of correct words \n',*corrected)
def ninja(list_words):
  array1=[]

  for q in range(len(list_words)):
      splits=wordninja.split(list_words[q])
    
      for i in range(len(splits)):
        if(len(splits[i])>2):
          array1.append(splits[i])
        
  print("Tokens after splitting strings:",*array1)
        
  return array1
list_words=ninja(list_words)
final_check=list(spell.unknown(list_words))
for i in list_words:
    if(i not in final_check):
        list_words.remove(i)
        
corrected.extend(list_words)
list_words=final_check

print('List of remaining words \n',*final_check)
print('List of correct words \n',*corrected)