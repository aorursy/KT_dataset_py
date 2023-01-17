# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install transformers -q
!pip install wandb -q
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
# Checking out the GPU we have access to. This is output is from the google colab version. 
!nvidia-smi
# # Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

# Preparing for TPU usage
# import torch_xla
# import torch_xla.core.xla_model as xm
# device = xm.xla_device()

print(device)
train_df = pd.read_json("/kaggle/input/naturalqa/nq-open-master/NQ-open.train.jsonl", orient='columns', lines=True)
train_df.head()
dev_df = pd.read_json("/kaggle/input/naturalqa/nq-open-master/NQ-open.dev.jsonl", orient='columns', lines=True)
dev_df.head()
number_of_answers = pd.Series([len(train_df['answer'][i]) for i in range(len(train_df['answer']))])
number_of_answers.value_counts()
train_df['number_of_answers']=number_of_answers
train_df.head()
sample_questions = train_df['question'].head()
sample_answers = train_df['answer'].head()
sample_answers
tokenizer = T5Tokenizer.from_pretrained("t5-base")
print(sample_questions[0])
tokenizer.encode(sample_questions[0])
sample_answers[4]
multi_answers = train_df.query('number_of_answers>1')['answer']
multi_answers.head()
answers = ' <sep> '.join(multi_answers[7])
answers
# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.question = self.data.question
        self.answer = self.data.answer

    def __len__(self):
        return len(self.question)

    def __getitem__(self, index):
        question = str(self.question[index])
        question = 'trivia question: '+' '.join(question.split())+'?'
        answer = ' <sep> '.join(self.answer[index])
        answer = ' '.join(answer.split())
        
        print(question,":",answer)
        
        source = self.tokenizer.batch_encode_plus([question], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([answer], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }
# Creating the training function. This will be called in the main function. It is run depending on the epoch value.
# The model is put into train mode and then we wnumerate over the training loader and passed to the defined network 

def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _,data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)
        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
        loss = outputs[0]

        if _%500==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # xm.optimizer_step(optimizer)
        # xm.mark_step()
def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
                
            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=150, 
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
            
            if _%100==0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals
TRAIN_BATCH_SIZE = 2    # input batch size for training (default: 64)
VALID_BATCH_SIZE = 2    # input batch size for testing (default: 1000)
TRAIN_EPOCHS = 2
# number of epochs to train (default: 10)
VAL_EPOCHS = 1 
LEARNING_RATE = 1e-4    # learning rate (default: 0.01)
SEED = 42               # random seed (default: 42)
MAX_LEN = 512

# Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(SEED) # pytorch random seed
np.random.seed(SEED) # numpy random seed
torch.backends.cudnn.deterministic = True

# tokenzier for encoding the text
tokenizer = T5Tokenizer.from_pretrained("t5-base")
# Creation of Dataset and Dataloader
# Defining the train size. So 80% of the data will be used for training and the rest will be used for validation. 
train_size = 0.95
train_df = pd.read_json("/kaggle/input/naturalqa/nq-open-master/NQ-open.train.jsonl", orient='columns', lines=True)
train_df = train_df[:10000]
train_dataset=train_df.sample(frac=train_size, random_state = SEED).reset_index(drop=True)
val_dataset=train_df.drop(train_dataset.index).reset_index(drop=True)
val_dataset = val_dataset
print("FULL Dataset: {}".format(train_df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(val_dataset.shape))
# Creating the Training and Validation dataset for further creation of Dataloader
training_set = CustomDataset(train_dataset[:20], tokenizer, MAX_LEN)
val_set = CustomDataset(val_dataset[:20], tokenizer, MAX_LEN)

# Defining the parameters for creation of dataloaders
train_params = {
    'batch_size': TRAIN_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0
    }

val_params = {
    'batch_size': VALID_BATCH_SIZE,
    'shuffle': False,
    'num_workers': 0
    }

# Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
training_loader = DataLoader(training_set, **train_params)
val_loader = DataLoader(val_set, **val_params)
for itm in training_loader:
    print()
for itm in val_loader:
    print()
# Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary. 
# Further this model is sent to device (GPU/TPU) for using the hardware.
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model = model.to(device)

# Defining the optimizer that will be used to tune the weights of the network in the training session. 
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
# Training loop
# print('Initiating Fine-Tuning for the model on our dataset')

# for epoch in range(TRAIN_EPOCHS):
#     train(epoch, tokenizer, model, device, training_loader, optimizer)
# Validation loop and saving the resulting file with predictions and acutals in a dataframe.
# Saving the dataframe as predictions.csv
# print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')
# predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
# final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
#  final_df.to_csv('predictions.csv')
# for i in range(20):
#     print("Actual Answer: ", final_df['Actual Text'][i],"\nPredicted Answer: ", final_df['Generated Text'][i])
#     print()
