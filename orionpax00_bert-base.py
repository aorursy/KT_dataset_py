import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os



all_csv_files_loc = []

all_csv_files_name = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename[-3:] != "txt" and filename not in all_csv_files_name:

            all_csv_files_loc.append(os.path.join(dirname, filename))

            all_csv_files_name.append(filename)

print("Total Unique hotels:", len(all_csv_files_name))



total_reviews = 0

for file in all_csv_files_loc:

    df = pd.read_csv(file)

    total_reviews += df.shape[0]

print("Total Reviews:", total_reviews)



total_reviews = 0

con_df = pd.read_csv(all_csv_files_loc[0]).values

for file in all_csv_files_loc[1:]:

    con_df = np.vstack((con_df, pd.read_csv(file).values))

print("Stacked Dataframe Shape:", con_df.shape)
#manipulating data

con_df = con_df[:,[1,5]]

con_df[:,1] /= 50

# con_df[:,1] -= 1
# Saving all reviews in a single file

pd.DataFrame(

    data=con_df,

    columns = ['review','rating']

).to_csv("./con_reviews.csv", index=False)
from tqdm import tqdm



import torch

import transformers



import torch.nn as nn

import torch.optim as optim
class config:

    MAX_LEN = 64

    TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    PRETRAINED = 'bert-base-uncased'

    

    DEVICE = torch.device("cuda")

    

    LOSS = nn.MSELoss()

    OPTIMIZER = None

    EPOCHS = 5

    BATCH_SIZE = 32

        
def process_data(review, target):

    tokens = config.TOKENIZER.tokenize(review)

    tokens = ['[CLS]'] + tokens + ['[SEP]']

    if len(tokens) < config.MAX_LEN:

        tokens = tokens + ['[PAD]' for _ in range(config.MAX_LEN - len(tokens))]

    else:

        tokens = tokens[:config.MAX_LEN - 1] + ['[SEP]']

        

    token_idx = torch.tensor(config.TOKENIZER.convert_tokens_to_ids(tokens))

    attention_mask = (token_idx != 0).long()

    

    return token_idx, attention_mask, target
class ReviewLoader:

    def __init__(self, review, target):

        self.review = review

        self.target = target

        

    def __len__(self):

        return len(self.review)

    

    def __getitem__(self, item):

        return process_data(

            self.review[item],

            self.target[item]

        )
class BertBASE(transformers.BertPreTrainedModel):

    def __init__(self, conf):

        super(BertBASE, self).__init__(conf)

        self.bert = transformers.BertModel.from_pretrained(config.PRETRAINED, config=conf)

        self.l0 = nn.Linear(768, 1)

        

    def forward(self, idx, mask):

        out = self.bert(

            idx,

            attention_mask=mask

        ) 

        logits = self.l0(out[0][:,0])

        

        return logits
def train_fn(data_loader, model):

    model.train()



    tk_data = tqdm(data_loader, total=len(data_loader))

    

    for it, (idx, mask, target) in enumerate(tk_data):



        idx = idx.to(config.DEVICE, dtype=torch.long)

        mask = mask.to(config.DEVICE, dtype=torch.long)

        target = target.to(config.DEVICE, dtype=torch.float)

        

        model.zero_grad()

        logits = model(

            idx = idx,

            mask = mask

        )

        

        losses = config.LOSS(logits.float(), target.resize_(config.BATCH_SIZE,1))

        losses.backward()

        config.OPTIMIZER.step()

        

        if it%200 == 0:

            print("LOSS:", losses.item())

    
def run():

    

    df = pd.read_csv('./con_reviews.csv')

    train_data_loader = torch.utils.data.DataLoader(

        ReviewLoader(

            review = df.review.values,

            target = df.rating.values

        ),

        batch_size = config.BATCH_SIZE,

        num_workers = 4

    )

    

    model_config = transformers.BertConfig.from_pretrained(config.PRETRAINED)

    model_config.output_hidden_states = True

    model = BertBASE(conf = model_config)

    model.to(config.DEVICE)

    

    config.OPTIMIZER = optim.Adam(model.parameters(), lr = 2e-5)

    

    for e in range(config.EPOCHS):

        train_fn(train_data_loader, model)
run()
torch.cuda.empty_cache()