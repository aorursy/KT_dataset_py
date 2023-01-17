import nltk
from nltk.tokenize import word_tokenize
filename = '../input/TheWonderfulWizardofOz_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words by white space
words = text.split()

# convert to lowercase
words = [word.lower() for word in words]

print('Manual approach words size:', len(words))
print(words[:100])

file = open('result_manual_approach.txt', 'w') 
file.write(str(words))
file.close() 
# split into words
tokens = word_tokenize(text)

print('NLTK approach tokens size:', len(tokens))

print(tokens[:100])
# convert to lower case
tokens = [w.lower() for w in tokens]

# remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in tokens]

# remove remaining tokens that are not alphabetic
words = [word for word in stripped if word.isalpha()]

# filter out stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]

print('NLTK approach tokens size:', len(words))

print(words[:100])

file = open('result_nltk_approach.txt', 'w') 
file.write(str(words))
file.close() 