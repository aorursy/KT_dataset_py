!ls ../input
import pprint
import json
from tqdm import tqdm
dataPath = "../input/train-v2.0.json"
with open(dataPath) as file:
    trainData = json.load(file)
type(trainData)
i=0
j = 0
total = 0
for article in trainData['data']:
    if i < 1:
        pprint.pprint(article)
        i+=1
    for para in article['paragraphs']:
        if j < 1:
            pprint.pprint(para)
            j+=1
        total += len(para['qas'])
total
len(trainData['data'])
import nltk
def tokenize(sequence):
    tokens = [token.replace("``", '"').replace("''", '"').lower() for token in nltk.word_tokenize(sequence)]
    return tokens
def get_char_word_loc_mapping(context, context_tokens):
    """
    Return a mapping that maps from character locations to the corresponding token locations.
    If we're unable to complete the mapping e.g. because of special characters, we return None.

    Inputs:
      context: string (unicode)
      context_tokens: list of strings (unicode)

    Returns:
      mapping: dictionary from ints (character locations) to (token, token_idx) pairs
        Only ints corresponding to non-space character locations are in the keys
        e.g. if context = "hello world" and context_tokens = ["hello", "world"] then
        0,1,2,3,4 are mapped to ("hello", 0) and 6,7,8,9,10 are mapped to ("world", 1)
    """
    acc = '' # accumulator
    current_token_idx = 0 # current word loc
    mapping = dict()

    for char_idx, char in enumerate(context): # step through original characters
        if char != u' ' and char != u'\n': # if it's not a space:
            acc += char # add to accumulator
            context_token = str(context_tokens[current_token_idx]) # current word token
            if acc == context_token: # if the accumulator now matches the current word token
                syn_start = char_idx - len(acc) + 1 # char loc of the start of this word
                for char_loc in range(syn_start, char_idx+1):
                    mapping[char_loc] = (acc, current_token_idx) # add to mapping
                acc = '' # reset accumulator
                current_token_idx += 1

    if current_token_idx != len(context_tokens):
        return None
    else:
        return mapping

num_exs = 0 # number of examples written to file
num_mappingprob, num_tokenprob, num_spanalignprob = 0, 0, 0
examples = []

i=0
j=0
k = 0
for articles_id in tqdm(range(len(trainData['data'])), 
                        desc="Preprocessing {}".format("train")):
    if i<5:
        print(articles_id)
        i+=1
    article_paragraphs = trainData['data'][articles_id]['paragraphs']
#     if j<1:
#         print(len(article_paragraphs))
#         print(article_paragraphs)
#         j+=1
    for pid in range(len(article_paragraphs)):
#         if j <1:
#             print(article_paragraphs[pid])
#             j+=1
        context = str(article_paragraphs[pid]['context'])
        context = context.replace("''", '" ')
        context = context.replace("``", '" ')
        
        context_tokens = tokenize(context) # list of strings (lowercase)
        context = context.lower()
        qas = article_paragraphs[pid]['qas'] # list of questions
        charloc2wordloc = get_char_word_loc_mapping(context, context_tokens)

#         if j<1:
#             print("printing the article_paragraph...")
#             pprint.pprint(article_paragraphs)
#             print("printing the context...")
#             pprint.pprint(context)
#             print("printing the qas...")
#             pprint.pprint(qas)
#             print("context tokens")
#             pprint.pprint(context_tokens)
#             print("printing charloc2wordloc...")
#             pprint.pprint(charloc2wordloc)
#             j+=1
        if charloc2wordloc is None: # there was a problem
            num_mappingprob += len(qas)
            continue # skip this context example

            
            
       
  # for each question, process the question and answer and write to file
        for qn in qas:

                   # read the question text and tokenize
            question = str(qn['question']) # string
            question_tokens = tokenize(question) # list of strings

            # of the three answers, just take the first
            pprint.pprint(qn['answers'][0]['text'].lower())
            print(qn['answers'][0]['text'])
            ans_text = qn['answers'][0]['text'].lower() # get the answer text
            ans_start_charloc = qn['answers'][0]['answer_start'] # answer start loc (character count)
            ans_end_charloc = ans_start_charloc + len(ans_text) # answer end loc (character count) (exclusive)
            
            if k<1:
                print("printing questions...")
                pprint.pprint(qn['answers'][0]['text'].lower())
#                 print("answer text...")
#                 pprint.pprint(ans_text)
                k+=1

        
