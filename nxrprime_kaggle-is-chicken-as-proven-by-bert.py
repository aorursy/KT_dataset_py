from transformers import *
import numpy as np
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

bert_model = TFBertForMaskedLM.from_pretrained('bert-base-uncased') 



# DEFINE SENTENCE

str = '[CLS] Data science is [MASK] . [SEP]'

indices = tokenizer.encode(str, add_special_tokens=False, return_tensors='tf')



# PREDICT MISSING WORDS

pred = bert_model(indices)

masked_indices = np.where(indices==103)[1]



# DISPLAY MISSING WORDS

predicted_words = np.argmax( np.asarray(pred[0][0])[masked_indices,:] ,axis=1)

print( tokenizer.decode(predicted_words) )
str = '[CLS] BERT is the [MASK] . [SEP]'

indices = tokenizer.encode(str, add_special_tokens=False, return_tensors='tf')



# PREDICT MISSING WORDS

pred = bert_model(indices)

masked_indices = np.where(indices==103)[1]



# DISPLAY MISSING WORDS

predicted_words = np.argmax( np.asarray(pred[0][0])[masked_indices,:] ,axis=1)

print( tokenizer.decode(predicted_words) )
str = '[CLS] Apple really [MASK] good those [MASK] . [SEP]'

indices = tokenizer.encode(str, add_special_tokens=False, return_tensors='tf')



# PREDICT MISSING WORDS

pred = bert_model(indices)

masked_indices = np.where(indices==103)[1]



# DISPLAY MISSING WORDS

predicted_words = np.argmax( np.asarray(pred[0][0])[masked_indices,:] ,axis=1)

print( tokenizer.decode(predicted_words) )
str = '[CLS] Kaggle is definitely not [MASK] . [SEP]'

indices = tokenizer.encode(str, add_special_tokens=False, return_tensors='tf')



# PREDICT MISSING WORDS

pred = bert_model(indices)

masked_indices = np.where(indices==103)[1]



# DISPLAY MISSING WORDS

predicted_words = np.argmax( np.asarray(pred[0][0])[masked_indices,:] ,axis=1)

print( tokenizer.decode(predicted_words) )
str = '[CLS] Google will dominate the [MASK] . [SEP]'

indices = tokenizer.encode(str, add_special_tokens=False, return_tensors='tf')



# PREDICT MISSING WORDS

pred = bert_model(indices)

masked_indices = np.where(indices==103)[1]



# DISPLAY MISSING WORDS

predicted_words = np.argmax( np.asarray(pred[0][0])[masked_indices,:] ,axis=1)

print( tokenizer.decode(predicted_words) )
str = '[CLS] Microsoft will languish in the [MASK] . [SEP]'

indices = tokenizer.encode(str, add_special_tokens=False, return_tensors='tf')



# PREDICT MISSING WORDS

pred = bert_model(indices)

masked_indices = np.where(indices==103)[1]



# DISPLAY MISSING WORDS

predicted_words = np.argmax( np.asarray(pred[0][0])[masked_indices,:] ,axis=1)

print( tokenizer.decode(predicted_words) )
str = '[CLS] The king of NLP will always [MASK] [MASK] . [SEP]'

indices = tokenizer.encode(str, add_special_tokens=False, return_tensors='tf')



# PREDICT MISSING WORDS

pred = bert_model(indices)

masked_indices = np.where(indices==103)[1]



# DISPLAY MISSING WORDS

predicted_words = np.argmax( np.asarray(pred[0][0])[masked_indices,:] ,axis=1)

print( tokenizer.decode(predicted_words) )
str = '[CLS] Kaggle is [MASK] . [SEP]'

indices = tokenizer.encode(str, add_special_tokens=False, return_tensors='tf')



# PREDICT MISSING WORDS

pred = bert_model(indices)

masked_indices = np.where(indices==103)[1]



# DISPLAY MISSING WORDS

predicted_words = np.argmax( np.asarray(pred[0][0])[masked_indices,:] ,axis=1)

print( tokenizer.decode(predicted_words) )
str = '[CLS] @philippsinger is a [MASK] grandmaster on Kaggle . [SEP]'

indices = tokenizer.encode(str, add_special_tokens=False, return_tensors='tf')



# PREDICT MISSING WORDS

pred = bert_model(indices)

masked_indices = np.where(indices==103)[1]



# DISPLAY MISSING WORDS

predicted_words = np.argmax( np.asarray(pred[0][0])[masked_indices,:] ,axis=1)

print( tokenizer.decode(predicted_words) )