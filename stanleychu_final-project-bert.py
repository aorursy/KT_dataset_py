!python -m pip install --upgrade pip

!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
import random

import pandas as pd

import numpy as np

from tqdm.notebook import tqdm

from sklearn.metrics import roc_auc_score

import IPython

import sys



import torch

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

import transformers

from transformers import BertForSequenceClassification, BertPreTrainedModel, BertConfig, BertModel

import torch.nn as nn

import torch.nn.functional as F



import torch_xla

import torch_xla.core.xla_model as xm
class config:

    EPOCHS = 1

    BATCH_SIZE = 32

    VAL_BATCH_SIZE = 128

    TEST_BATCH_SIZE = 128

    LR = 3e-5
valid = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation-processed-seqlen128.csv")

train = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train-processed-seqlen128.csv")

test = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test-processed-seqlen128.csv")

submit = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")

train = train[['id', 'comment_text', 'input_word_ids', 'input_mask','all_segment_id', 'toxic']].iloc[:20000]
class TweetDataset(Dataset):

    def __init__(self, mode, df):

        self.mode = mode

        self.df = df

    

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        token, segment, mask = self.df.loc[idx, ["input_word_ids", "all_segment_id", "input_mask"]].values

        if self.mode=="train" or self.mode == "valid":

            label_tensor = torch.tensor(self.df.loc[idx, "toxic"])

        else:

            label_tensor = torch.tensor(-1)

        tokens_tensor = torch.tensor([int(i) for i in token[1:-1].split(",")])

        segments_tensor = torch.tensor([int(i) for i in segment[1:-1].split(",")])

        masks_tensor = torch.tensor([int(i) for i in mask[1:-1].split(",")])

           

        return tokens_tensor, segments_tensor, masks_tensor, label_tensor
lang = {'Spanish': 'es', 'Italian': 'it', 'Turkish': 'tr'}



validsets = {}

for i, k in lang.items():

    validsets[i] = TweetDataset("valid", valid[valid["lang"] == k].reset_index(drop=True))

trainset = TweetDataset("train", train)

validset = TweetDataset("valid", valid)

testset = TweetDataset("test", test)



validloaders = {}

for i, k in validsets.items():

    validloaders[i] = DataLoader(k, batch_size=config.VAL_BATCH_SIZE, num_workers=4, shuffle=False)

trainloader = DataLoader(trainset, batch_size=config.BATCH_SIZE, num_workers=4, shuffle=False)

validloader = DataLoader(validset, batch_size=config.VAL_BATCH_SIZE, num_workers=4, shuffle=False)

testloader = DataLoader(testset, batch_size=config.TEST_BATCH_SIZE, num_workers=4, shuffle=False)
class Model(nn.Module):

    

    def __init__(self, labels=1):

        

        super().__init__()

        

        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")

        self.num_features = self.bert.pooler.dense.out_features

        self.labels = labels

        

        self.drop = nn.Dropout(0.3)

        self.fc1 = nn.Linear(self.num_features * 2, self.num_features)

        self.logit = nn.Linear(self.num_features, self.labels)

        

    def forward(self, tokens_tensors, segments_tensors, masks_tensors):

        

        hidden_states, cls = self.bert(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors)

        avgpool = torch.mean(hidden_states, 1)

        maxpool, _ = torch.max(hidden_states, 1)

        cat = torch.cat((avgpool, maxpool), 1)

        x = self.drop(cat)

        x = torch.tanh(self.fc1(x))

        output = self.logit(x)



        return output
model = Model()
model
device = xm.xla_device()

model.to(device)

print(f"Now we use {device}\n")
def training(model, warmup_prop=0.1):



    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)

    num_warmup_steps = int(warmup_prop * config.EPOCHS * len(trainloader))

    num_training_steps = config.EPOCHS * len(trainloader)

    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    loss_fun = torch.nn.BCEWithLogitsLoss(reduction='mean').to(device)    



    for epoch in range(config.EPOCHS):

        model.train()

        

        optimizer.zero_grad()

        avg_loss = 0

        

        for data in tqdm(trainloader):             

            tokens_tensor, segments_tensor, masks_tensor, labels_tensor = [k.to(device) for k in data if k is not None]

            output = model(tokens_tensor, segments_tensor, masks_tensor)

            loss = loss_fun(output.view(-1).float(), labels_tensor.float().to(device))

            loss.backward()

            avg_loss += loss.item() / len(trainloader)



            xm.optimizer_step(optimizer, barrier=True)

            scheduler.step()

            model.zero_grad()

            optimizer.zero_grad()

                

        model.eval()

        preds = []

        truths = []

        avg_val_loss = 0.



        with torch.no_grad():

            for data in validloader:

                tokens_tensor, segments_tensor, masks_tensor, labels_tensor = [k.to(device) for k in data if k is not None]

                output = model(tokens_tensor, segments_tensor, masks_tensor)

                loss = loss_fun(output.detach().view(-1).float(), labels_tensor.float().to(device))

                avg_val_loss += loss.item() / len(validloader)

                

                probs = torch.sigmoid(output).detach().cpu().numpy()

                preds += list(probs.flatten())

                truths += list(labels_tensor.detach().cpu().numpy().flatten())

            score = roc_auc_score(truths, preds)

        

        lr = scheduler.get_last_lr()[0]

        print(f'[Epoch {epoch + 1}] lr={lr:.1e} loss={avg_loss:.4f} val_loss={avg_val_loss:.4f} val_auc={score:.4f}')
threshold = lambda x: 1 if x>=0.5 else 0



def predict(model, dataloader, df, isAccuracy=True):

 

    model.eval().to(device)

    preds = np.empty((0, 1))

    accuracy = None



    with torch.no_grad():

        for data in tqdm(dataloader):

            tokens_tensor, segments_tensor, masks_tensor, labels_tensor = [k.to(device) for k in data if k is not None]

            probs = torch.sigmoid(model(tokens_tensor, segments_tensor, masks_tensor)).detach().cpu().numpy()

            preds = np.concatenate([preds, probs])

            

    preds = preds.reshape(len(preds))        

    predicts = np.array([threshold(i) for i in preds])

    if isAccuracy:

        accuracy = (df["toxic"].values == predicts).sum() / len(df)



    return preds, predicts, accuracy 
# before training model accuracy

pre, pre_class, accuracy = predict(model, trainloader, train)

auc = roc_auc_score(train["toxic"].values, pre_class)

print("Train: ")

print(f"Model before fine-tune accuracy: {accuracy * 100:.3f}%\nModel before fine-tune AUC: {auc:.3f}")



for key, value in validloaders.items():

    pre, pre_class, accuracy = predict(model, value, valid[valid["lang"] == lang[key]].reset_index(drop=True))

    auc = roc_auc_score(valid[valid["lang"] == lang[key]].reset_index(drop=True)["toxic"].values, pre_class)

    print(f"{key} Valid: ")

    print(f"Model before fine-tune accuracy: {accuracy * 100:.2f}%\nModel before fine-tune AUC: {auc:.3f}")



pre, pre_class, accuracy = predict(model, validloader, valid)

auc = roc_auc_score(valid["toxic"].values, pre_class)

print(f"Combined Valid: ")

print(f"Model before fine-tune accuracy: {accuracy * 100:.2f}%\nModel before fine-tune AUC: {auc:.3f}")
%%time 



training(model)
# After training model accuracy

pre, pre_class, accuracy = predict(model, trainloader, train)

auc = roc_auc_score(train["toxic"].values, pre)

print("Train: ")

print(f"Model before fine-tune accuracy: {accuracy * 100:.3f}%\nModel before fine-tune AUC: {auc:.3f}")



for key, value in validloaders.items():

    pre, pre_class, accuracy = predict(model, value, valid[valid["lang"] == lang[key]].reset_index(drop=True))

    auc = roc_auc_score(valid[valid["lang"] == lang[key]].reset_index(drop=True)["toxic"].values, pre)

    print(f"{key} Valid: ")

    print(f"Model before fine-tune accuracy: {accuracy * 100:.2f}%\nModel before fine-tune AUC: {auc:.3f}")



pre, pre_class, accuracy = predict(model, validloader, valid)

auc = roc_auc_score(valid["toxic"].values, pre)

print(f"Combined Valid: ")

print(f"Model before fine-tune accuracy: {accuracy * 100:.2f}%\nModel before fine-tune AUC: {auc:.3f}")
torch.save(model.state_dict(), "./model.bin")
pre, pre_class, accuracy = predict(model, testloader, test, False)

submit['toxic'] = pre

submit.to_csv('submission.csv', index=False)

submit.head()
!test -d bertviz_repo || git clone https://github.com/jessevig/bertviz bertviz_repo

if not 'bertviz_repo' in sys.path:

    sys.path += ['bertviz_repo']



from transformers import BertTokenizer, BertModel

from bertviz import head_view



def call_html():

    display(IPython.core.display.HTML('''

        <script src="/static/components/requirejs/require.js"></script>

        <script>

          requirejs.config({

            paths: {

              base: '/static/base',

              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",

              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',

            },

          });

        </script>

        '''))

model_version = 'bert-base-chinese'

model = BertModel.from_pretrained(model_version, output_attentions=True)

tokenizer = BertTokenizer.from_pretrained(model_version)



sentence_a = "阿明去買東西，"

sentence_b = "回來的時候要給他錢。"



inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)

token_type_ids = inputs['token_type_ids']

input_ids = inputs['input_ids']

attention = model(input_ids, token_type_ids=token_type_ids)[-1]

input_id_list = input_ids[0].tolist() # Batch index 0

tokens = tokenizer.convert_ids_to_tokens(input_id_list)

call_html()



head_view(attention, tokens)