def lower_text(text):
  text = text.lower()
  return text
text = "India is the second most populated country after China and the fourth largest economy after China  other two are USA, Russia."
output_str = lower_text(text)
print(output_str)
import re

def remove_nembers(text):
  input_str = re.sub(r'\d+', "", input)
  return input_str
text = "There are 20 stairs for the first floor while 21 for the second floor but in total there are 44 stairs how ?"
input_str = remove_nembers(text)
input_str
def remove_Punctuation(string): 
  
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for x in string.lower(): 
        if x in punctuations: 
            string = string.replace(x, "") 
    return string
input_str = "This &is [an] example? {of} string. with.? punctuation!!!!" # Sample string
output_str =  remove_Punctuation(input_str) 
output_str
def remove_whitespace(input_str):
    input_str = input_str.strip()
    
    return input_str
text = " \t a string example\t "
output_str = remove_whitespace(text)
output_str

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_stopwords(text):
  stop_words = set(stopwords.words('english'))
  tokens = word_tokenize(text)
  result = [i for i in tokens if not i in stop_words]
  return " ".join(result)

inp_str = "NLTK is a leading platform for building Python programs to work with human language data."
opt_str = remove_stopwords(inp_str)
print(opt_str)
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def stemming(text):
  tokens = []
  stemmer= PorterStemmer()
  tokenize_word=word_tokenize(text)
  for word in tokenize_word:
      tokens.append(stemmer.stem(word))
  return " ".join(tokens)
input_str="There are several types of stemming algorithms."
stemming(input_str) 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('wordnet')

def lemmatization(text):
  tokens = []
  lemmatizer=WordNetLemmatizer()
  input_str=word_tokenize(text)
  for word in input_str:
      tokens.append(lemmatizer.lemmatize(word))
  return " ".join(tokens)
text="been had done languages cities mice"
output_str = lemmatization(text)
output_str
import re

def cleaning_html_tags(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext
text = "The tag is <html> </html> removed"
cleaning_html_tags(text)
import re

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
text  = "https://github.com/KnitVikas"
output_utls = remove_urls(text)
output_utls
text = "text does not https://github.com/KnitVikas coantain any urls"
output_text = remove_urls(text)
output_text # so this is still removing only the urls fro the text
import re

def remove_emojis(text):
  print(text) # with emoji
  emoji_pattern = re.compile("["
          u"\U0001F600-\U0001F64F"  # emoticons
          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
          u"\U0001F680-\U0001F6FF"  # transport & map symbols
          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)
  return emoji_pattern.sub(r'', text)
text = u'This dog \U0001f602'
output_text = remove_emojis(text)
output_text
def remove_emoticons(text):
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)
remove_emoticons("Hello :-)")
def convert_emojis(text):
    for emot in UNICODE_EMO:
        text = re.sub(r'('+emot+')', "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()), text)
    return text
text = "game is on 🔥"
convert_emojis(text)
def convert_emoticons(text):
    for emot in EMOTICONS:
        text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
    return text
text = "Hello :-) :-)"
convert_emoticons(text)
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize 
from nltk import pos_tag


def pos_tagging(text):
  text = word_tokenize(text)
  return nltk.pos_tag(text)
text = "Parts of speech examples: an article, to write, interesting, easily, and, of"
output_text = pos_tagging(text)
print(output_text)
import spacy 
nlp = spacy.load('en_core_web_sm') 

def named_entity_recognition(text):
  doc = nlp(text) 
  text = ""  
  for ent in doc.ents: 
    text = text + "" + ent.text + "--> " + ent.label_ +" "
  return text
text = "Apple is looking at buying U.K. startup for $1 billion"
output_str = named_entity_recognition(text)
output_str
# !pip install symspellpy
import pkg_resources
from symspellpy import SymSpell, Verbosity

def spellcorrector(text):
  sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
  dictionary_path = pkg_resources.resource_filename(
      "symspellpy", "frequency_dictionary_en_82_765.txt")
  bigram_path = pkg_resources.resource_filename(
      "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
  # term_index is the column of the term and count_index is the
  # column of the term frequency
  sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
  sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
  # lookup suggestions for multi-word input strings (supports compound
  # splitting & merging)
  # max edit distance per lookup (per single word, not per whole input string)
  suggestions = sym_spell.lookup_compound((text), max_edit_distance=2)
  # display suggestion term, edit distance, and term frequency
  corrected_text = ""
  for suggestion in suggestions:
      corrected_text =corrected_text + str(suggestion).split(",")[0]

  return corrected_text
wrong_text = "whereis th elove hehad dated forImuch of thepast who couqdn'tread in sixtgrade and ins pired him"
correct_text = spellcorrector(wrong_text)
correct_text
import requests

def chat_words_conversion(slangText):
  prefixStr = '<div class="translation-text">'
  postfixStr = '</div'
  r = requests.post('https://www.noslang.com/', {'action': 'translate', 'p': 
  slangText, 'noswear': 'noswear', 'submit': 'Translate'})
  startIndex = r.text.find(prefixStr)+len(prefixStr)
  endIndex = startIndex + r.text[startIndex:].find(postfixStr)
  return r.text[startIndex:endIndex]
Text = "one minute BRB"
chat_words_conversion(Text)