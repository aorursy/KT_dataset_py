# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install transformers
import torch
from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
question = "How many parameters does BERT-large have?"
answer_text = "BERT-large is really big... it has 24-layers and an embedding size of 1,024, for a total of 340M parameters! Altogether it is 1.34GB, so expect it to take a couple minutes to download to your Colab instance."
input_ids = tokenizer.encode(question, answer_text)

print('The input has a total of {:} tokens.'.format(len(input_ids)))
tokens = tokenizer.convert_ids_to_tokens(input_ids)

# For each token and its id...
for token, id in zip(tokens, input_ids):
    
    # If this is the [SEP] token, add some space around it to make it stand out.
    if id == tokenizer.sep_token_id:
        print('')
    
    # Print the token string and its ID in two columns.
    print('{:<12} {:>6,}'.format(token, id))

    if id == tokenizer.sep_token_id:
        print('')
sep_index = input_ids.index(tokenizer.sep_token_id)

# The number of segment A tokens includes the [SEP] token istelf.
num_seg_a = sep_index + 1

# The remainder are segment B.
num_seg_b = len(input_ids) - num_seg_a

# Construct the list of 0s and 1s.
segment_ids = [0]*num_seg_a + [1]*num_seg_b
assert len(segment_ids) == len(input_ids)
start_scores, end_scores = model(torch.tensor([input_ids]),token_type_ids=torch.tensor([segment_ids]))
answer_start = torch.argmax(start_scores)
answer_end = torch.argmax(end_scores)
# Combine the tokens in the answer and print it out.
answer = ' '.join(tokens[answer_start:answer_end+1])

print('Answer: "' + answer + '"')
answer = tokens[answer_start]
for i in range(answer_start + 1, answer_end + 1):
    
    # If it's a subword token, then recombine it with the previous token.
    if tokens[i][0:2] == '##':
        answer += tokens[i][2:]
    
    # Otherwise, add a space then the token.
    else:
        answer += ' ' + tokens[i]

print('Answer: "' + answer + '"')
import matplotlib.pyplot as plt
import seaborn as sns

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
#sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (16,8)
s_scores = start_scores.detach().numpy().flatten()
e_scores = end_scores.detach().numpy().flatten()
# We'll use the tokens as the x-axis labels. In order to do that, they all need
# to be unique, so we'll add the token index to the end of each one.
token_labels = []
for (i, token) in enumerate(tokens):
    token_labels.append('{:} - {:>2}'.format(token, i))

ax = sns.barplot(x=token_labels, y=s_scores, ci=None)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
ax.grid(True)

plt.title('Start Word Scores')

plt.show()
# Create a barplot showing the end word score for all of the tokens.
ax = sns.barplot(x=token_labels, y=e_scores, ci=None)

# Turn the xlabels vertical.
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

# Turn on the vertical grid to help align words to scores.
ax.grid(True)

plt.title('End Word Scores')

plt.show()
def answer_question(question, answer_text):
    
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text)
    encoded_dict = tokenizer.encode_plus(text=question,text_pair=answer_text, add_special=True)
    segment_ids = encoded_dict['token_type_ids']
    #print(token_type_ids)
    # Report how long the input sequence is.
    print('Query has {:,} tokens.\n'.format(len(input_ids)))

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
#     sep_index = input_ids.index(tokenizer.sep_token_id)

#     # The number of segment A tokens includes the [SEP] token istelf.
#     num_seg_a = sep_index + 1

#     # The remainder are segment B.
#     num_seg_b = len(input_ids) - num_seg_a

#     # Construct the list of 0s and 1s.
#     segment_ids = [0]*num_seg_a + [1]*num_seg_b
    #print(segment_ids)
    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example question through the model.
    print(torch.tensor([input_ids]))
    print(torch.tensor([segment_ids]))
    start_scores, end_scores = model(torch.tensor([input_ids],dtype=torch.long),token_type_ids=torch.tensor([segment_ids],dtype=torch.long))

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
        
        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        
        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    print('Answer: "' + answer + '"')
import textwrap
import nltk
# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=100) 

bert_abstract = """The incubation period of the virus is the time between the exposure and the display of symptoms. Current information suggests that the incubation period ranges from 1 to 12.5 days (with median estimates of 5 to 6 days), but can be as long as 14 days.
Older people, and people with pre-existing medical conditions (such as asthma, diabetes, heart disease) appear to be more and can be infected by the new coronavirus . Coronaviruses are a large family of viruses that are known to cause illness ranging from the common cold to more severe diseases such as Middle East Respiratory Syndrome (MERS) and Severe Acute Respiratory Syndrome (SARS). 
Detailed investigations found that SARS-CoV was transmitted from civet cats to humans in China in 2002 and MERS-CoV from dromedary camels to humans in Saudi Arabia in 2012. Several known coronaviruses are circulating in animals that have not yet infected humans. As surveillance improves around the world, more coronaviruses are likely to be identified. 
It depends on the virus, but common signs include respiratory symptoms, fever, cough, shortness of breath, and breathing difficulties. In more severe cases, infection can cause pneumonia, severe acute respiratory syndrome, kidney failure and even death.
Yes, some coronaviruses can be transmitted from person to person, usually after close contact with an infected patient, for example, in a household workplace, or health care centre. When a disease is new, there is no vaccine until one is developed. It can take a number of years for a new vaccine to be developed. There is no specific treatment for disease caused by a novel coronavirus. 
However, many of the symptoms can be treated and therefore treatment based on the patient’s clinical condition. Moreover, supportive care for infected persons can be highly effective. Standard recommendations to reduce exposure to and transmission of a range of illnesses include maintaining basic hand and respiratory hygiene, and safe food practices  and avoiding close contact, when possible, with anyone showing symptoms of respiratory illness such as coughing and sneezing. 
Kabir Das was a 15th-century Indian mystic poet and saint, whose writings influenced Hinduism's Bhakti movement and his verses are found in Sikhism's scripture Guru Granth Sahib. His early life was in a Muslim family, but he was strongly influenced by his teacher, the Hindu bhakti leader Ramananda.
Yes, health care workers come into contact with patients more often than the general public WHO recommends that health care workers consistently apply appropriate. WHO encourages all countries to enhance their surveillance for severe acute respiratory infections (SARI), to carefully review any unusual patterns of SARI or pneumonia cases and to notify WHO of any suspected or confirmed case of infection with novel coronavirus."""
print(wrapper.fill(bert_abstract))
question = "What is a coronavirus?"

answer_question(question, bert_abstract)
question = "What are the symptoms of someone infected with a coronavirus?"

answer_question(question, bert_abstract)
question = "What is a corona virus?"

answer_question(question, bert_abstract)
question = "what is the date of birth of the President of India?"

answer_question(question, bert_abstract)
question = "Is there a treatment for a novel coronavirus?"
answer_question(question, bert_abstract)