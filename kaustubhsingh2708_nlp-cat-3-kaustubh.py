#Question1
#Done by Kaustubh
f = open("/kaggle/input/nlpcat3/CAT3.txt", "r")
print(f.read())

#Assigning a variable with smaller text from the given data to perform operations
f = open("/kaggle/input/nlpcat3/CAT3.txt", "r")
lines_to_read = [0,10,11]

file1 = open("myfile.txt", "w")
for position, line in enumerate(f):
    if position in lines_to_read:
        file1.write(str(line))
        file1.write("\n")
file1.close()
file1 = open("myfile.txt","r") 
data=file1.read()
file1.close()
print(data)


#Question 2
#Removing Stop words
#done by Kaustubh

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
  

  
stop_words = set(stopwords.words('english')) 
  
word_tokens = word_tokenize(data) 
  
filtered = [w for w in word_tokens if not w in stop_words] 
  
filtered = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered.append(w) 
print("Sentence Before filtration\n")  
print(word_tokens,'\n') 
print("Sentence after filtration\n")
print(filtered_sentence) 

#Question 3
#Tokenizing the given text
#Done by Kaustubh

import nltk.data 
from nltk.tokenize import word_tokenize   

tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle') 
#line tokenize
print("Line tokenize\n\n")
print(tokenizer.tokenize(data)) 
#word tokenize
print("\n\n Word tokenize\n\n")
word_token=word_tokenize(data)
print(word_token)
#Question 4
#Assign part of speech tag for each token
#done by Rupam
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
token = nltk.word_tokenize(data)
pos = nltk.pos_tag(token)
print(pos)