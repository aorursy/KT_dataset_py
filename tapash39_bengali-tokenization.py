!pip install bnlp-toolkit
from bnlp.basic_tokenizer import BasicTokenizer

basic_tokenizer = BasicTokenizer(False) # Here False means do_lower_case=False

raw_text = "আমি বাংলায় গান গাই।"

tokens = basic_tokenizer.tokenize(raw_text)

print(tokens)
from bnlp.nltk_tokenizer import NLTK_Tokenizer



text = "আমি ভাত খাই। সে বাজারে যায়। তিনি কি সত্যিই ভালো মানুষ?"

bengali_nltk = NLTK_Tokenizer(text)

word_tokens = bengali_nltk.word_tokenize()

sentence_tokens = bengali_nltk.sentence_tokenize()

print("Word Tokens: ", word_tokens)

print("Sentence Tokens: ", sentence_tokens)
from bnlp.sentencepiece_tokenizer import SP_Tokenizer





bengali_sentencepiece = SP_Tokenizer()

model_path = "/kaggle/input/bn_spm.model"

input_text = "আমি ভাত খাই। সে বাজারে যায়।"

tokens = bengali_sentencepiece.tokenize(model_path, input_text)

print(tokens)

text2id = bengali_sentencepiece.text2id(model_path, input_text)

print(text2id)

id2text = bengali_sentencepiece.id2text(model_path, text2id)

print(id2text)