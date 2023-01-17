!pip install torchvision
!pip uninstall -y transformers

!pip install transformers
import numpy as np

import torch

from transformers import BertTokenizer, BertForMaskedLM
BERT_MODEL = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

model = BertForMaskedLM.from_pretrained(BERT_MODEL)
def get_masks(tokens, max_len=128):

    """Mask for padding"""

    if len(tokens) > max_len:

        raise IndexError("Token length more than max length!")

    return [1] * len(tokens) + [0] * (max_len - len(tokens))





def get_segments(tokens, max_len=128):

    """Segments: 0 for the first sequence, 1 for the second"""

    if len(tokens) > max_len:

        raise IndexError("Token length more than max length!")

    segments = []

    current_segment_id = 0

    for token in tokens:

        segments.append(current_segment_id)

        if token == "[SEP]":

            current_segment_id = 1

    return segments + [0] * (max_len - len(tokens))





def get_ids(tokens, tokenizer, max_len=128):

    """Token ids from Tokenizer vocab"""

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_ids = token_ids + [tokenizer.pad_token_id] * (max_len - len(token_ids))

    return input_ids





def encode_input(sentence, max_len=128):

    stokens = tokenize(sentence)

    

    input_ids = get_ids(stokens, tokenizer, max_len)

    input_masks = get_masks(stokens, max_len)

    input_segments = get_segments(stokens, max_len)

    

    model_input = {

        'input_word_ids': np.array([input_ids]),

        'input_masks': np.array([input_masks]),

        'input_segments': np.array([input_segments])

    }

    

    mask_pos = np.array(np.array(stokens) == '[MASK]', dtype='int')

    mask_pos = np.concatenate((mask_pos, np.zeros(max_len - len(mask_pos))))

    mask_pos = mask_pos.astype('int')

    

    return model_input, mask_pos
def tokenize(sentence):

    stokens = tokenizer.tokenize(sentence)

    i = 0

    while i < len(stokens) - 2:

        if stokens[i] == '[' and stokens[i+1] == 'mask' and stokens[i+2] == ']':

            stokens[i] = '[MASK]'

            stokens.pop(i+2)

            stokens.pop(i+1)

        i = i + 1

    

    stokens = ['[CLS]'] + stokens + ['[SEP]']

    return stokens
text = "Hello, I'm a [MASK] model."

tokenize(text)
# Example of BERT inputs

text = "Hello, I'm a [MASK] model."

encode_input(text, max_len=15) # max_len=15 for display purpose
text = "Hello, I'm a [MASK] model."

tokenizer.tokenize(text)
text = "Hello, I'm a [MASK] model."



encoded_input = tokenizer(text, return_tensors='pt')

encoded_input
def get_topk_predictions(model, tokenizer, text, topk=5):

    encoded_input = tokenizer(text, return_tensors='pt')

    logits = model(encoded_input['input_ids'],

                   encoded_input['token_type_ids'],

                   encoded_input['attention_mask'],

                   masked_lm_labels=None)[0]



    logits = logits.squeeze(0)

    probs = torch.softmax(logits, dim=-1)



    mask_cnt = 0

    token_ids = encoded_input['input_ids'][0]

    

    top_preds = []



    for idx, _ in enumerate(token_ids):

        if token_ids[idx] == tokenizer.mask_token_id:

            mask_cnt += 1

            

            topk_prob, topk_indices = torch.topk(probs[idx, :], topk)

            topk_indices = topk_indices.cpu().numpy()

            topk_tokens = tokenizer.convert_ids_to_tokens(topk_indices)

            for prob, tok_str, tok_id in zip(topk_prob, topk_tokens, topk_indices):

                top_preds.append({'token_str': tok_str,

                                  'token_id': tok_id,

                                  'probability': float(prob)})

    

    return top_preds
def display_topk_predictions(model, tokenizer, text, pretty_prob=False):

    top_preds = get_topk_predictions(model, tokenizer, text)

    

    print(text)

    print('=' * 40)

    for item in top_preds:

        if not pretty_prob:

            print('%s %.4f' % (item['token_str'], item['probability']))

        else:

            probability = item['probability'] * 100

            print('%s %.2f%%' % (item['token_str'], probability))
text = "Hello, I'm a [MASK] model."

display_topk_predictions(model, tokenizer, text)
text = 'The doctor ran to the emergency room to see [MASK] patient.'

display_topk_predictions(model, tokenizer, text)
text = 'Este coche es [MASK].'

display_topk_predictions(model, tokenizer, text)
text = 'Este coche es muy [MASK].'

display_topk_predictions(model, tokenizer, text)
text = 'Я считаю, что Настя очень [MASK] человек.'

display_topk_predictions(model, tokenizer, text, pretty_prob=True)