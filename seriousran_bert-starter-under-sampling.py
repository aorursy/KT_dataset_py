import gc

import os

import time

import math

import random

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from datetime import date

from transformers import *

from sklearn.metrics import *

from tqdm.notebook import tqdm



import torch

import torch.nn as nn

import torch.utils.data

import torch.nn.functional as F



from torch import Tensor

from torch.optim import *

from torch.nn.modules.loss import *

from torch.optim.lr_scheduler import * 

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import RandomSampler
def seed_everything(seed):

    """

    Seeds basic parameters for reproductibility of results

    

    Arguments:

        seed {int} -- Number of the seed

    """

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False
seed = 2020

seed_everything(seed)
MODEL_PATHS = {

    'bert-multi-cased': '../input/bertconfigs/multi_cased_L-12_H-768_A-12/multi_cased_L-12_H-768_A-12/',

}
DATA_PATH = '../input/jigsaw-multilingual-toxic-comment-classification/'



df_val = pd.read_csv(DATA_PATH + 'validation-processed-seqlen128.csv')

df_test =  pd.read_csv(DATA_PATH + 'test-processed-seqlen128.csv')

df_train = pd.read_csv(DATA_PATH + 'jigsaw-toxic-comment-train-processed-seqlen128.csv')

df_train_2 = pd.read_csv(DATA_PATH + 'jigsaw-unintended-bias-train-processed-seqlen128.csv')
%%time

print('mean of severe_toxicity', np.mean(df_train_2[df_train_2['severe_toxicity']!=0]['severe_toxicity'].values))

print('mean of obscene', np.mean(df_train_2[df_train_2['obscene']!=0]['obscene'].values))

print('mean of identity_attack', np.mean(df_train_2[df_train_2['identity_attack']!=0]['identity_attack'].values))

print('mean of insult', np.mean(df_train_2[df_train_2['insult']!=0]['insult'].values))

print('mean of threat', np.mean(df_train_2[df_train_2['threat']!=0]['threat'].values))
df_train_2
print(len(df_train_2[df_train_2['severe_toxicity']>0.5]))

print(len(df_train_2[df_train_2['obscene']>0.8]))

print(len(df_train_2[df_train_2['identity_attack']>0.7]))

print(len(df_train_2[df_train_2['insult']>0.9]))

print(len(df_train_2[df_train_2['threat']>0.7]))
plt.hist(df_train_2[df_train_2['toxic']>0]['toxic'])
df_train_2['toxic'] = df_train_2['toxic'].fillna(0).round().astype(int)

df_train_2['severe_toxic'] = df_train_2['severe_toxicity'].fillna(0).round().astype(int)

df_train_2['obscene'] = df_train_2['obscene'].fillna(0).round().astype(int)

df_train_2['identity_hate'] = df_train_2['identity_attack'].fillna(0).round().astype(int)

df_train_2['insult'] = df_train_2['insult'].fillna(0).round().astype(int)

df_train_2['threat'] = df_train_2['threat'].fillna(0).round().astype(int)

df_train_2.head()
# %%time

# for i, row in df_train_2.iterrows():

#     if row['toxic'] < 0.6:

#         df_train_2.at[i,'toxic'] = 0

#         continue

#     df_train_2.at[i,'toxic'] = 1

#     if row['severe_toxicity']>0.5:

#         df_train_2.at[i,'severe_toxic'] = 1

#     else:

#         df_train_2.at[i,'severe_toxic'] = 0

#     if row['obscene']>0.5:

#         df_train_2.at[i,'obscene'] = 1

#     else:

#         df_train_2.at[i,'obscene'] = 0

#     if row['identity_attack']>0.5:

#         df_train_2.at[i,'identity_hate'] = 1

#     else:

#         df_train_2.at[i,'identity_hate'] = 0

#     if row['insult']>0.5:

#         df_train_2.at[i,'insult'] = 1

#     else:

#         df_train_2.at[i,'insult'] = 0

#     if row['threat']>0.5:

#         df_train_2.at[i,'threat'] = 1

#     else:

#         df_train_2.at[i,'threat'] = 0

        

# df_train_2['toxic'].astype(int)

# df_train_2['severe_toxic'].astype(int)

# df_train_2['obscene'].astype(int)

# df_train_2['identity_hate'].astype(int)

# df_train_2['insult'].astype(int)

# df_train_2['threat'].astype(int)
df_train_2 = df_train_2[df_train.columns]

df_train_2.head()
df_train.head()
sns.countplot(df_train['toxic'])

plt.title('Target repartition on training data')

plt.show()
df_train.head()
df_train.toxic.value_counts()[1]
count_least = min(df_train.severe_toxic.value_counts()[1],

                  df_train.obscene.value_counts()[1],

                  df_train.threat.value_counts()[1],

                  df_train.insult.value_counts()[1],

                  df_train.identity_hate.value_counts()[1])

print(count_least)
df_train_types = []

for toxic_type in ['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:

    df_class_0 = df_train[df_train[toxic_type] == 0]

    df_class_1 = df_train[df_train[toxic_type] == 1]



    df_class_0_under = df_class_0.sample(count_least)

    df_class_1_under = df_class_1.sample(count_least)

    

    df_train_types.append(df_class_0_under)

    df_train_types.append(df_class_1_under)

        

df_train_under = pd.concat(df_train_types, axis=0)

print(df_train_under.shape)
sns.countplot(df_train_under['toxic'])

plt.title('Target repartition on training data')

plt.show()
sns.countplot(df_train_under['severe_toxic'])

plt.title('Target repartition on training data')

plt.show()
sns.countplot(df_train_under['obscene'])

plt.title('Target repartition on training data')

plt.show()
sns.countplot(df_train_under['threat'])

plt.title('Target repartition on training data')

plt.show()
sns.countplot(df_train_under['insult'])

plt.title('Target repartition on training data')

plt.show()
sns.countplot(df_train_under['identity_hate'])

plt.title('Target repartition on training data')

plt.show()
count_0 = df_train_under.toxic.value_counts()[0]

count_1 = df_train_under.toxic.value_counts()[1]



df_class_0 = df_train_2[df_train_2['toxic'] == 0]

df_class_1 = df_train_2[df_train_2['toxic'] == 1]



df_class_0_under = df_class_0.sample(count_1)

df_class_1_under = df_class_1.sample(count_0)



df_train_under = pd.concat([df_train_under, df_class_0_under, df_class_1_under], axis=0)
sns.countplot(df_train_under['toxic'])

plt.title('Target repartition on training data')

plt.show()
class JigsawDataset(Dataset):

    """

    Torch dataset for training and validating

    """

    def __init__(self, df):

        super().__init__()

        self.df = df 

        self.word_ids = np.array([word_ids[1:-1].split(', ') for word_ids in df['input_word_ids']]).astype(int)

        

        try:

            self.y = df['toxic'].values

        except KeyError: # test data

            self.y = np.zeros(len(df))



    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        return torch.tensor(self.word_ids[idx]), torch.tensor(self.y[idx])
TRANSFORMERS = {

    "bert-multi-cased": (BertModel, BertTokenizer, "bert-base-cased"),

}
model_class, tokenizer_class, pretrained_weights = TRANSFORMERS["bert-multi-cased"]

bert_config = BertConfig.from_json_file(MODEL_PATHS["bert-multi-cased"] + 'bert_config.json')
class Transformer(nn.Module):

    def __init__(self, model, num_classes=1):

        """

        Constructor

        

        Arguments:

            model {string} -- Transformer to build the model on. Expects "camembert-base".

            num_classes {int} -- Number of classes (default: {1})

        """

        super().__init__()

        self.name = model



        model_class, tokenizer_class, pretrained_weights = TRANSFORMERS[model]



        bert_config = BertConfig.from_json_file(MODEL_PATHS[model] + 'bert_config.json')

        bert_config.output_hidden_states = True

        

        bert_config.max_position_embeddings = 128

        bert_config.hidden_dropout_prob = 0.0

        

        

        self.transformer = BertModel(bert_config)



        self.nb_features = self.transformer.pooler.dense.out_features



        self.pooler = nn.Sequential(

            nn.Linear(self.nb_features, self.nb_features), 

            nn.Tanh(),

        )



        self.logit = nn.Linear(self.nb_features, num_classes)



    def forward(self, tokens):

        """

        Usual torch forward function

        

        Arguments:

            tokens {torch tensor} -- Sentence tokens

        

        Returns:

            torch tensor -- Class logits

        """

        _, _, hidden_states = self.transformer(

            tokens, attention_mask=(tokens > 0).long()

        )



        hidden_states = hidden_states[-1][:, 0] # Use the representation of the first token of the last layer



        ft = self.pooler(hidden_states)



        return self.logit(ft)
def fit(model, train_dataset, val_dataset, epochs=1, batch_size=8, warmup_prop=0, lr=5e-4):

    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



    optimizer = AdamW(model.parameters(), lr=lr)

    

    num_warmup_steps = int(warmup_prop * epochs * len(train_loader))

    num_training_steps = epochs * len(train_loader)

    

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)



    loss_fct = nn.BCEWithLogitsLoss(reduction='mean').cuda()

    

    for epoch in range(epochs):

        model.train()

        start_time = time.time()

        

        optimizer.zero_grad()

        avg_loss = 0

        

        for step, (x, y_batch) in enumerate(train_loader): 

            y_pred = model(x.cuda())

            

            loss = loss_fct(y_pred.view(-1).float(), y_batch.float().cuda())

            loss.backward()

            avg_loss += loss.item() / len(train_loader)



            optimizer.step()

            scheduler.step()

            model.zero_grad()

            optimizer.zero_grad()

                

        model.eval()

        preds = []

        truths = []

        avg_val_loss = 0.



        with torch.no_grad():

            for x, y_batch in val_loader:                

                y_pred = model(x.cuda())

                loss = loss_fct(y_pred.detach().view(-1).float(), y_batch.float().cuda())

                avg_val_loss += loss.item() / len(val_loader)

                

                probs = torch.sigmoid(y_pred).detach().cpu().numpy()

                preds += list(probs.flatten())

                truths += list(y_batch.numpy().flatten())

            score = roc_auc_score(truths, preds)

            

        

        dt = time.time() - start_time

        lr = scheduler.get_last_lr()[0]

        print(f'Epoch {epoch + 1}/{epochs} \t lr={lr:.1e} \t t={dt:.0f}s \t loss={avg_loss:.4f} \t val_loss={avg_val_loss:.4f} \t val_auc={score:.4f}')
epochs = 2

batch_size = 32

warmup_prop = 0.2 

lr = 1e-4
model = Transformer("bert-multi-cased").cuda()
n = len(df_train_under)  # I do not train on the entier data as it will take too long (for now)

print('the length of train data is', n)

#train_dataset = JigsawDataset(df_train_under.sample(n))

train_dataset = JigsawDataset(df_train_under)
val_dataset = JigsawDataset(df_val)

test_dataset = JigsawDataset(df_test)
fit(model, train_dataset, val_dataset, epochs=epochs, batch_size=batch_size, warmup_prop=warmup_prop, lr=lr)
def predict(model, dataset, batch_size=64):

    """

    Usual predict torch function

    

    Arguments:

        model {torch model} -- Model to predict with

        dataset {torch dataset} -- Dataset to get predictions from

    

    Keyword Arguments:

        batch_size {int} -- Batch size (default: {32})

    

    Returns:

        numpy array -- Predictions

    """



    model.eval()

    preds = np.empty((0, 1))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)



    with torch.no_grad():

        for x, _ in tqdm(loader):

            probs = torch.sigmoid(model(x.cuda())).detach().cpu().numpy()

            preds = np.concatenate([preds, probs])

            

    return preds
pred_val = predict(model, val_dataset)
score = roc_auc_score(df_val['toxic'], pred_val)

print(f'Scored {score:.4f} on validation data')
pred_test = predict(model, test_dataset)
sub = pd.read_csv(DATA_PATH + 'sample_submission.csv')

sub['toxic'] = pred_test

sub.to_csv('submission.csv', index=False)

sub.head()