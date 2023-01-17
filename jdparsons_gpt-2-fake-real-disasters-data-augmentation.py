import os

import sys

import torch

import random

import numpy as np

from nltk.tokenize import word_tokenize

import numpy as np

import pandas as pd

import time



# From https://github.com/graykode/gpt-2-Pytorch couldn't find a pip version

# I uploaded this gpt-2-Pytorch library as a dataset, so it would permanently

# reside in the input folder, which allowed the notebook commit sucessfully.

os.chdir('/kaggle/input/gpt2pytorch/gpt-2-Pytorch')

sys.path.insert(1, '/kaggle/input/gpt2pytorch/gpt-2-Pytorch/')



from GPT2.model import (GPT2LMHeadModel)

from GPT2.utils import load_weight

from GPT2.config import GPT2Config

from GPT2.sample import sample_sequence

from GPT2.encoder import get_encoder



# set pandas preview to use full width of browser

pd.set_option('display.max_columns', None)

pd.set_option('display.expand_frame_repr', False)

pd.set_option('max_colwidth', -1)

view_local_files = False



if view_local_files is True:

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            print(os.path.join(dirname, filename))
# https://www.kaggle.com/bkkaggle/generate-your-own-text-with-openai-s-gpt-2-117m



state_dict = torch.load(

    '../../../input/gpt2pytorch-modelbin/gpt2-pytorch_model.bin',

    map_location='cpu' if not torch.cuda.is_available() else None)



seed = random.randint(0, 2147483647)

np.random.seed(seed)

torch.random.manual_seed(seed)

torch.cuda.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Load Model

enc = get_encoder()

config = GPT2Config()

model = GPT2LMHeadModel(config)

model = load_weight(model, state_dict)

model.to(device)

model.eval()



def force_period(text):

    """If input string does not end with common punctuation,

    a period is added to the end. credit:

    https://stackoverflow.com/a/41402588

    A dangling word at the end of a sentence that doesn't

    end with punctuation causes GPT-2 to go off topic.

    """

    if text[-1] not in ['!', ',', '.', '\n']:

        text += '.'

    

    return text



def clean(text):

    """Removes various characters and string patterns

    generated by GPT-2.

    """

    text = text.replace('\n', ' ').replace('<|endoftext|>', '').strip()

    

    return text



def text_generator(state_dict,

                   text,

                   match_length=True,

                   match_length_multiplier=2,

                   length=50,

                   temperature=0.5,

                   top_k=30):

    """code by TaeHwan Jung(@graykode)

    Original Paper and repository here : https://github.com/openai/gpt-2

    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT

    Modifications by John David Parsons for the Kaggle "Real or Not?" competition

    Depends on external GPT2 variables initialized outside of this function.

    

    Args:

        text: sentence to begin with.

        length: number of words to generate, only read if match_length is False

        temperature: 0=deterministic, 1.0 is wildly creative and risks going off topic

        

    Returns:

        A string of GPT-2 generated text, based on the input text.

    """

    

    text = force_period(text)

    

    # very short texts benefit from a longer multiplier

    if len(text) < 30:

        match_length_multiplier += 1

    

    # very long texts do not need as much multiplier

    if len(text) > 120 and match_length_multiplier > 1:

        match_length_multiplier -= 1

    

    if match_length is True:

        length = len(word_tokenize(text)) * match_length_multiplier



    # max tweet length is 280 characters, estimating a max of 50 words

    length = min(length, 50)

    unconditional = False



    context_tokens = enc.encode(text)



    out = sample_sequence(

        model=model,

        length=length,

        context=context_tokens if not unconditional else None,

        start_token=enc.encoder['<|endoftext|>'] if unconditional else None,

        batch_size=1,

        temperature=temperature,

        top_k=top_k,

        device=device)

    out = out[:, len(context_tokens):].tolist()



    text = enc.decode(out[0])

    text = clean(text)



    return text





def get_fake_tweets(df, num_samples=10):

    """Generates fake text similar to the original. NOTE:

    enabling the GPU will speed up execution by about 2x.

    60 rows took 85 seconds on a CPU, 45 seconds on a GPU



    Args:

        df: A pandas dataframe with columns 'text' and 'target'

        num_samples: number of rows to generate



    Returns:

        A pandas dataframe containing only the new generated

        text. The dataframe has the following columns:

        'original_text', 'fake_text', 'target'

    """

    

    start_time = time.time()

    expanded_rows = []



    for i, row in df.sample(num_samples).iterrows():

        row_original_text = row['text']

        row_target = row['target']



        generated_text = text_generator(state_dict, row_original_text)

        expanded_row = [row_original_text, generated_text, row_target]

        expanded_rows.append(expanded_row)



    print("--- %s seconds ---" % (time.time() - start_time))



    expanded_df = pd.DataFrame(

        expanded_rows, columns=['original_text', 'fake_text', 'target'])



    return expanded_df
#train_df = pd.read_csv('../../../input/nlp-getting-started/train.csv')

train_df = pd.read_csv('../../../input/tweet-cleaner/train_df_clean.csv')



train_df = train_df[['text', 'target']]



train_df
# iloc of interesting test tweets

# 5725 = rescuing bodies in the water

# 333 = Windows is ethics armageddon

# 5678 = Dog buried alive

# 7611 = e-bike crash

# 7 = fire in the woods



test_tweet = train_df.iloc[7611]['text']

#test_tweet = 'Wow, it is super stormy out right now. The lightning woke me up :/'

generated_tweet = text_generator(state_dict, test_tweet, match_length_multiplier=2)



print('ORIGINAL: ' + test_tweet)

print('GPT-2: ' + generated_tweet)
# num_samples=3000 took around 40 min with the GPU

faked_df = get_fake_tweets(train_df, num_samples=10)

faked_df = faked_df[['fake_text', 'target']]

faked_df.columns = ['text', 'target']

faked_df.to_csv('../../../working/faked_df.csv', index=False)

faked_df
train_df_combined = pd.concat([train_df, faked_df])

train_df_combined.to_csv('../../../working/train_df_combined.csv', index=False)