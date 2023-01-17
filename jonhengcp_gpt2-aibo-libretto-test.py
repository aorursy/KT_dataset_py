!pip install gpt-2-finetuning==1.0.14

!pip install tensorflow==1.15
import os

import tensorflow as tf

import numpy as np



from tqdm import tqdm

from gpt_2_finetuning.conditional_sample_model import ConditionalSampleModel
tf.test.is_gpu_available()
!ls ../input
checkpoint_dir = "../input/gpt2-aibo-30000-v2"

# libretto_file = "../input/aibo-libretto/LibrettoEva2.txt"

libretto_file = "../input/libretto-eva-v3/LibrettoEva3.txt"
with open(libretto_file, 'r') as f:

    libretto_text = f.readlines()
libretto_text[:5]
len(libretto_text)
class LibrettoBot:

    MAX_HISTORY = 10

    CHAT_SEP = '\n\n'



    def __init__(self, model):

        self.model = model

        self.actor_name = 'EVA'

        self.bot_name = 'BOT'

        self.chat_history = []



    def display_history(self):

        print('=========================')

        print(self.history_as_string())

        print('=========================')



    def history_as_string(self):

        return self.CHAT_SEP.join(self.chat_history)



    def actor_prompt(self, prompt):

        self.chat_history.append(self.actor_name + '\n' + prompt)

        self.chat_history.append(self.bot_name + '\n')



        while len(self.chat_history) > self.MAX_HISTORY:

            self.chat_history.pop(0)



        bot_reply = self.model.run(self.history_as_string())[0]

        bot_reply = self.postprocess_reply(bot_reply)



        self.chat_history[-1] = self.bot_name + '\n' + bot_reply



    def postprocess_reply(self, reply_str):

        eol_chars = '!.?'

        reply_str = reply_str.replace('\t', '')

        reply_str = reply_str.replace('\n', '')

        reply_str = reply_str.lstrip().rstrip()



        truncated_str = ''

        for char in reply_str:

            if char not in eol_chars:

                truncated_str += char

            else:

                truncated_str += char

                break



        return truncated_str



    def clear_history(self):

        self.chat_history = []



    def get_last_response(self):

        return self.chat_history[-1][len(self.bot_name) + 1:]
# model = ConditionalSampleModel(checkpoint_dir, sample_length=50, temperature=2.0)

# bot = LibrettoBot(model)



# # loop to generate responses

# for prompt in tqdm(libretto_text[:3]):

#     prompt = prompt.replace('\n', '')

#     bot.actor_prompt(prompt)

# bot.display_history()
# bot.chat_history

# bot.clear_history()
## Write bot responses only

# with open('bot_responses.txt', 'w') as f:

#     for prompt in tqdm(libretto_text):

#         prompt = prompt.replace('\n', '')

#         bot.actor_prompt(prompt)

#         f.write(bot.get_last_response() + '\n')
# ## Write chat history

# number_of_runs = 5

# for i in range(1, number_of_runs + 1):

#     with open(f'bot_responses_{i}.txt', 'w') as f:

#         for prompt in tqdm(libretto_text):

#             prompt = prompt.replace('\n', '')

#             bot.actor_prompt(prompt)

#             f.write(bot.chat_history[-2] + '\n\n')

#             f.write(bot.chat_history[-1] + '\n\n')
def generate_dialogue(n_runs, output_prefix, temperature):

    model = ConditionalSampleModel(checkpoint_dir, sample_length=50, temperature=temperature)

    bot = LibrettoBot(model)

    

    for i in range(1, n_runs + 1):

        with open(f'{output_prefix}_{i}.txt', 'w') as f:

            for prompt in tqdm(libretto_text):

                prompt = prompt.replace('\n', '')

                bot.actor_prompt(prompt)

                f.write(bot.chat_history[-2] + '\n\n')

                f.write(bot.chat_history[-1] + '\n\n')
generate_dialogue(n_runs=3, output_prefix='responses_temp1_v2', temperature=1.0)

generate_dialogue(n_runs=3, output_prefix='responses_temp1.2_v2', temperature=1.2)

generate_dialogue(n_runs=3, output_prefix='responses_temp1.5_v2', temperature=1.5)

generate_dialogue(n_runs=3, output_prefix='responses_temp2_v2', temperature=2.0)

generate_dialogue(n_runs=3, output_prefix='responses_temp3_v2', temperature=3.0)