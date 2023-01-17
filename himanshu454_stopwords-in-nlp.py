import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
print(stopwords.words('english'))
len( stopwords.words('english'))
sample_text = "Oh man, this is pretty cool. We will do more such things."
text_tokens = word_tokenize(sample_text)
tokens_without_sw = [word for word in text_tokens if not word in set(stopwords.words('english'))]

print("Original text : " , text_tokens)
print("Remove StopWords : " , tokens_without_sw)