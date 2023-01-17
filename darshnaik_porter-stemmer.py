"""Applying PorterStmmer from nltk package"""



from nltk.stem import PorterStemmer

from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

new_ls = ["likes","liked","likely","liking","programs","programer","programing","undo","friendly"]

for i in new_ls:

    print(i + "  ->  "+ps.stem(i))