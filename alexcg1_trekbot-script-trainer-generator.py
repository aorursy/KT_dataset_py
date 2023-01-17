import torch

from torch.utils.data import Dataset

from torch.utils.data import Dataset, DataLoader

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelWithLMHead

import numpy as np

import os

import random

from datetime import datetime

from IPython.display import clear_output

from time import sleep

from zipfile import ZipFile

import subprocess

from utils import *
os.chdir("/kaggle/working") # Make sure we're in base directory
# How many episodes to train on? If 0, assume all episodes

EP_COUNT = 0



# How many times to cycle through all episodes in training?

EPOCHS = 1



# How many batches to run together?

BATCH_SIZE = 1



# Testing string. Trekbot starts with this to create a new script

# SAMPLE_STRING = "Picard to Riker"



# Sample length for samples generated from model

SAMPLE_LENGTH = 500



# What text should we train on? Default is Star Trek episodes

# TRAINING_FILE_PATH = os.path.join("..", "input", "trekbot", "film_text.txt")
# with open(TRAINING_FILE_PATH, "r") as file:

#     print(file.read()[0:400])
# MODEL_NAME = 'trekbot'

# MODEL_DIR = os.path.join("..", "input", "trekbot-model")



FILENAME_SUFFIX = str(datetime.now())[:10]



# Filename for output scripts

SCRIPT_DIR = f"scripts/"

SCRIPT_FILENAME = f"scripts-{FILENAME_SUFFIX}.txt"



# Directory to save model

OUTPUT_MODEL_DIR = f"models/trekbot"

OUTPUT_MODEL_FILENAME = f"model.txt"



# This variable already used in code later

output_dir = OUTPUT_MODEL_DIR
dirs = ['models', 'scripts', SCRIPT_DIR, 'samples', 'diagnostics']



Setup.dir_setup(dirs) # Setup directory structure
device = 'cpu'

if torch.cuda.is_available():

    device = 'cuda'
MODEL_NAME = 'alexcg1/trekbot'



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loaded Tokenizer from {MODEL_NAME}")

model = AutoModelWithLMHead.from_pretrained(MODEL_NAME)

print(f"Loaded Model from {MODEL_NAME}")
model = model.to(device)
import pickle

pickle_path = os.path.join("..", "input", "trekbot", "gpt2_trekbot.txt")

dataset = pickle.load(open(pickle_path, "rb"))

print(f"Dataset has {len(dataset)} scripts total")
if EP_COUNT != 0:

    dataset = dataset[:EP_COUNT]
script_loader = DataLoader(dataset,batch_size=1,shuffle=True)

if EP_COUNT != 0:

    print(f"Loaded {EP_COUNT} scripts from dataset")

else:

    print("Loaded all scripts from dataset")
LEARNING_RATE = 0.00006 # Faster uses more GPU?

WARMUP_STEPS = 10000
model = model.to(device)

model.train()

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=-1)

script_count = 0

sum_loss = 0.0

batch_count = 0
total = len(dataset) # number of items in dataset
from math import floor



SAMPLE_INTERVAL = 200

total_sample_count = floor(total/SAMPLE_INTERVAL)

print(f"Total samples: {total_sample_count}")
start_time = datetime.now().strftime("%H:%M:%S")

print(f"Start: {start_time}")
# Setup diagnostics

losses = []

loss_log = "loss_rate.csv"



Diag.setup(loss_log)
os.chdir(SCRIPT_DIR)
# %%timeit

for epoch in range(EPOCHS):

#     print(f"EPOCH {epoch} started" + '=' * 30)

    for idx,script in enumerate(script_loader):

                

        outputs = model(script.to(device), labels=script.to(device))

        

        loss, logits = outputs[:2]

        timestamp = datetime.now().strftime("%H:%M:%S")

        

        loss.backward()



        sum_loss = sum_loss + loss.detach().data

        basic_loss = loss.detach().data

        losses.append(basic_loss)

                       

        script_count = script_count + 1

        if script_count == BATCH_SIZE:

            script_count = 0    

            batch_count += 1

            optimizer.step()

            scheduler.step() 

            optimizer.zero_grad()

            model.zero_grad()

        

        if batch_count == SAMPLE_INTERVAL:

            append_string = f"\t| Appending script to {SCRIPT_FILENAME}..."

        else:

            append_string = ''

        

        # Update output display

        clear_output(wait=True) # Clear and update display, otherwise endless scroll

        percent = round(idx/total, 5)*100

        rounded_loss = round(float(loss.detach().data), 4)

        rounded_sum_loss = round(float(sum_loss), 2)

        print(f"{timestamp}: Processing {idx+1}/{total} \t {percent}% \tLoss: {rounded_loss} | Sum loss: {rounded_sum_loss} {append_string}")

        if batch_count == SAMPLE_INTERVAL:

            model.eval()

#             losses.append(sum_loss)

            sample_outputs = model.generate(

                                    bos_token_id=random.randint(1,30000),

                                    do_sample=True,   

                                    top_k=50, 

                                    max_length = SAMPLE_LENGTH,

                                    top_p=0.95, 

                                    num_return_sequences=1

                                )



            for i, sample_output in enumerate(sample_outputs):

                with open(SCRIPT_FILENAME, "a") as file:

                    file.write("\n\n")

                    file.write("*" * 80)

                    file.write("\n\n")

                    file.write(tokenizer.decode(sample_output, skip_special_tokens=True))

                    print(f"Script appended to {SCRIPT_FILENAME}")

            

            batch_count = 0

            sum_loss = 0.0

            model.train()
end_time = datetime.now().strftime("%H:%M:%S")

print(f"Start: {start_time}")

print(f"End: {end_time}")
os.chdir("/kaggle/working")

os.mkdir(OUTPUT_MODEL_DIR)



from transformers import WEIGHTS_NAME, CONFIG_NAME

output_model_file = os.path.join(OUTPUT_MODEL_DIR, WEIGHTS_NAME)

output_config_file = os.path.join(OUTPUT_MODEL_DIR, CONFIG_NAME)



torch.save(model.state_dict(), output_model_file)

print(f"Saved {output_model_file} to {OUTPUT_MODEL_DIR}")

model.config.to_json_file(output_config_file)

print(f"Saved {output_config_file} to {OUTPUT_MODEL_DIR}")

tokenizer.save_vocabulary(OUTPUT_MODEL_DIR)

print(f"Saved vocabulary to {OUTPUT_MODEL_DIR}")
model = GPT2LMHeadModel.from_pretrained(output_dir)

tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

print(f"Loaded model and tokenizer from {output_dir}")
sample_outputs = model.generate(

    bos_token_id=random.randint(1,30000),

    do_sample=True,   

    top_k=50, 

    max_length = 1000,

    top_p=0.95, 

    num_return_sequences=5

    )
# Generate samples

for i, output in enumerate(sample_outputs):

    filename = f'script_{i+1:03}.txt'

    file_path = f'samples/{filename}'

    content = tokenizer.decode(output, skip_special_tokens=True)

    with open(file_path, 'w') as file:

        file.write(content)

    print(f"{filename} written")

    
# Zip samples

os.chdir('samples')

with ZipFile('samples.zip', 'w') as zipObj:

    for filename in os.listdir():

        if not filename.endswith(".zip"):

            zipObj.write(filename)



print("samples.zip created")
print(f"Start loss {losses[0]}")

print(f"End loss {losses[-1]}")





loss_floats = []

for loss in losses:

    loss = float(loss)

    loss_floats.append(loss)
from matplotlib import pyplot as plt

plt.plot(loss_floats)

plt.show()