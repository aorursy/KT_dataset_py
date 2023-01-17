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
from transformers import AutoTokenizer, AutoModelWithLMHead



tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")



model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-large")
from transformers import AutoModelWithLMHead, AutoTokenizer

import torch





tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")

model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-large")



# Let's chat for 5 lines

for step in range(5):

    # encode the new user input, add the eos_token and return a tensor in Pytorch

    new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')



    # append the new user input tokens to the chat history

    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids



    # generated a response while limiting the total chat history to 1000 tokens, 

    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)



    # pretty print last ouput tokens from bot

    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))