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
import tensorflow as tf



# Get the GPU device name.

device_name = tf.test.gpu_device_name()



# The device name should look like the following:

if device_name == '/device:GPU:0':

    print('Found GPU at: {}'.format(device_name))

else:

    raise SystemError('GPU device not found')

import torch



# If there's a GPU available...

if torch.cuda.is_available():    



    # Tell PyTorch to use the GPU.    

    device = torch.device("cuda")



    print('There are %d GPU(s) available.' % torch.cuda.device_count())



    print('We will use the GPU:', torch.cuda.get_device_name(0))



# If not...

else:

    print('No GPU available, using the CPU instead.')

    device = torch.device("cpu")
!pip install transformers
from transformers import BertForQuestionAnswering, AdamW, BertConfig, RobertaForQuestionAnswering, RobertaTokenizer, BertTokenizer





output_dir = '/kaggle/input/roberta-2'











# # Load a trained model and vocabulary that you have fine-tuned

model = RobertaForQuestionAnswering.from_pretrained(output_dir)

tokenizer = RobertaTokenizer.from_pretrained(output_dir)



# # Copy the model to the GPU.

model.to(device)
import pandas as pd

import numpy as np

from torch.utils.data import TensorDataset, random_split

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler



max_len = 192





# Load the dataset into a pandas dataframe.

df_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")



# Report the number of sentences.

print('Number of test sentences: {:,}\n'.format(df_test.shape[0]))

df_test['id_num'] = np.arange(len(df_test))



# Tokenize all of the sentences and map the tokens to thier word IDs.

input_ids = []

attention_masks = []

token_type_ids = []

textID = []



# For every sentence...

for i in range(len(df_test['text'])):

    question = 'what portion of texts best reflect ' + df_test['sentiment'][i] + 'sentiment'

    # `encode_plus` will:

    #   (1) Tokenize the sentence.

    #   (2) Prepend the `[CLS]` token to the start.

    #   (3) Append the `[SEP]` token to the end.

    #   (4) Map tokens to their IDs.

    #   (5) Pad or truncate the sentence to `max_length`

    #   (6) Create attention masks for [PAD] tokens.

    text = df_test['text'][i]

    encoded_dict = tokenizer.encode_plus(

                        question,

                        text,                      # Sentence to encode.

                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'

                        max_length = max_len,           # Pad & truncate all sentences.

                        return_token_type_ids = True,

                        pad_to_max_length = True,

                        return_attention_mask = True,   # Construct attn. masks.

                        return_tensors = 'pt',     # Return pytorch tensors.

                   )

    

    # Add the encoded sentence to the list.    

    input_ids.append(encoded_dict['input_ids'])

    

    # And its attention mask (simply differentiates padding from non-padding).

    attention_masks.append(encoded_dict['attention_mask'])



    # Add its token type map

    token_type_ids.append(encoded_dict['token_type_ids'])



    textID.append(df_test['id_num'][i])





# Convert the lists into tensors.

input_ids = torch.cat(input_ids, dim=0)

attention_masks = torch.cat(attention_masks, dim=0)

token_type_ids = torch.cat(token_type_ids, dim=0)

textID = torch.tensor([int(x) for x in textID])





# Set the batch size.  

batch_size = 32  



# Create the DataLoader.



prediction_data = TensorDataset(input_ids, attention_masks,  token_type_ids, textID)

prediction_sampler = SequentialSampler(prediction_data)

prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
def id_to_word(answer_start, answer_end, input_ids):

    idx = input_ids[int(answer_start)+1:int(answer_end)+1]

    answer = tokenizer.decode(idx)

    return str(answer)





def get_start_end(start_score, end_score):

    starts = np.zeros(len(start_score))

    ends = np.zeros(len(start_score))

    for i in range(len(start_score)):

        total_score = []

        arg = []

        for a in range(len(start_score[i])):

            for b in range(a, len(end_score[i])):

                total_score.append( start_score[i][a] + end_score[i][b])

                arg.append((a,b))



        total_score = torch.tensor(total_score)



        max_idx = torch.argmax(total_score)

        start, end = arg[max_idx]

        starts[i] = start

        ends[i] = end

    return starts, ends
predictions  = []

textID = []

# Predict 

for batch in prediction_dataloader:

  # Add batch to GPU

    batch = tuple(t.to(device) for t in batch)

  

  # Unpack the inputs from our dataloader

    b_input_ids, b_input_mask, b_input_type_ids, b_textID = batch

  

  # Telling the model not to compute or store gradients, saving memory and 

  # speeding up prediction

    with torch.no_grad():

      # Forward pass, calculate logit predictions

        start_score, end_score, hidden_state = model(b_input_ids, 

                                   token_type_ids = b_input_type_ids,

                                   attention_mask=b_input_mask

                                   )



        # Move logits and labels to CPU

        start_score_pred = start_score.detach().cpu().numpy()

        end_score_pred = end_score.detach().cpu().numpy()



        b_input_ids = b_input_ids.to('cpu').numpy()

        b_textID = b_textID.detach().cpu().numpy()

        # Calculate the accuracy for this batch of test sentences, and

        # accumulate it over all batches.

        start_pred, end_pred = get_start_end(start_score_pred, end_score_pred)

        for i in range(len(b_input_ids)):

            predicted_text = tokenizer.decode(b_input_ids[i][int(start_pred[i]):int(end_pred[i])]) 

            predictions.append(predicted_text)

            textID.append(b_textID[i])



df = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")

df['selected_text'] = predictions

# df['selected_text'] = df['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split())==1 else x)

# df['selected_text'] = df['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split())==1 else x)

# df['selected_text'] = df['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split())==1 else x)



df.to_csv('/kaggle/working/submission.csv', index=False)

df.head(20)