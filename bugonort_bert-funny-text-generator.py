from transformers import BertTokenizer, TFBertForMaskedLM

import numpy as np



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

bert_model = TFBertForMaskedLM.from_pretrained('bert-base-uncased') 
# text = input('Write your text here: ') # Just start and type your phrase # Uncomment this line

text = 'Some text' # Comment this line!!! It only for committing notebook
num_of_words = np.random.randint(3,8) # How many new words generate



all_text = [text]

sentence = ('[CLS] {} [MASK] . [SEP]'.format(text))

for i in range(num_of_words):

    

    indices = tokenizer.encode(sentence, add_special_tokens=False, return_tensors='tf')

    prediction = bert_model(indices)

    masked_indices = np.where(indices==103)[1]



    output = np.argmax( np.asarray(prediction[0][0])[masked_indices,:] ,axis=1)

    new_word = tokenizer.decode(output)

    all_text.append(' ' + new_word)

    new_text = ''.join(all_text)

    sentence = ('[CLS] {} [MASK] . [SEP]'.format(new_text))



print(''.join(all_text))