%%capture

!pip install ../input/sacremoses/sacremoses-master/ > /dev/null



import os

import sys

import glob

import torch



sys.path.insert(0, "../input/transformers/transformers-master/")

!pip install /kaggle/input/transformers/transformers-2.2.1-py3-none-any.whl
!mkdir -p ./xlnet-base-cased

!cp /kaggle/input/xlnetbasecased/xlnet-base-cased-config.json ./xlnet-base-cased/config.json

!cp /kaggle/input/xlnetbasecased/xlnet-base-cased-pytorch_model.bin ./xlnet-base-cased/pytorch_model.bin

!cp /kaggle/input/xlnetbasecased/xlnet-base-cased-spiece.model ./xlnet-base-cased/spiece.model
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, DataLoader

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

import os

import math

from torch.nn import BCEWithLogitsLoss

from torch import nn

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import AdamW, XLNetTokenizer, XLNetModel, XLNetLMHeadModel, XLNetConfig

from tqdm.notebook import tqdm

from tqdm import trange

import matplotlib.pyplot as plt

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_gpu = torch.cuda.device_count()

torch.cuda.get_device_name(0)

df_test  = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')

df_test['host_topic'] = df_test.host.apply(lambda x: x.split('.')[0])

df_train = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')

df_train['host_topic'] = df_train.host.apply(lambda x: x.split('.')[0])

df_submit = pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')
target_cols = ['question_asker_intent_understanding',

       'question_body_critical', 'question_conversational',

       'question_expect_short_answer', 'question_fact_seeking',

       'question_has_commonly_accepted_answer',

       'question_interestingness_others', 'question_interestingness_self',

       'question_multi_intent', 'question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written', 'answer_helpful',

       'answer_level_of_information', 'answer_plausible', 'answer_relevance',

       'answer_satisfaction', 'answer_type_instructions',

       'answer_type_procedure', 'answer_type_reason_explanation',

       'answer_well_written']
plt.figure()

fig, ax = plt.subplots(figsize=(20, 10));

df_train[target_cols].hist(ax=ax);

plt.tight_layout()

plt.show()
def tokenize_inputs(text_list, tokenizer, num_embeddings=512, cut_class = True):

    """

    Tokenizes the input text input into ids. Appends the appropriate special

    characters to the end of the text to denote end of sentence. Truncate or pad

    the appropriate sequence length.

    original author: Josh Xin Jie Lee @ Medium

    """

    # tokenize the text, then truncate sequence to the desired length minus 2 for

    cut = 1

    if not cut_class:

        cut += 1

    # the 2 special characters

    tokenized_texts = [tokenizer.tokenize(t)[:num_embeddings-cut] for t in text_list]

    # convert tokenized text into numeric ids for the appropriate LM

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    # append special token "<s>" and </s> to end of sentence

    if cut_class:

        input_ids = [tokenizer.build_inputs_with_special_tokens(x)[:-1] for x in input_ids]

    else:

        input_ids = [tokenizer.build_inputs_with_special_tokens(x) for x in input_ids]

    # print(input_ids)

    # input()

    # pad sequences

    input_ids = pad_sequences(input_ids, maxlen=num_embeddings, dtype="long", truncating="post", padding="post")

    return input_ids
tokenizer = XLNetTokenizer.from_pretrained('./xlnet-base-cased/', do_lower_case=True)
def tokenize_inputs(text_list, tokenizer, num_embeddings=512, cut_class = True):

    """

    Tokenizes the input text input into ids. Appends the appropriate special

    characters to the end of the text to denote end of sentence. Truncate or pad

    the appropriate sequence length.

    original author: Josh Xin Jie Lee @ Medium

    """

    # tokenize the text, then truncate sequence to the desired length minus 2 for

    cut = 1

    if not cut_class:

        cut += 1

    # the 2 special characters

    tokenized_texts = [tokenizer.tokenize(t)[:num_embeddings-cut] for t in text_list]

    # convert tokenized text into numeric ids for the appropriate LM

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    # append special token "<s>" and </s> to end of sentence

    if cut_class:

        input_ids = [tokenizer.build_inputs_with_special_tokens(x)[:-1] for x in input_ids]

    else:

        input_ids = [tokenizer.build_inputs_with_special_tokens(x) for x in input_ids]

    # print(input_ids)

    # input()

    # pad sequences

    input_ids = pad_sequences(input_ids, maxlen=num_embeddings, dtype="long", truncating="post", padding="post")

    return input_ids
%%time

T = tokenize_inputs(df_train.question_title.values, tokenizer, 64)

Q = tokenize_inputs(df_train.question_body.values, tokenizer, 224)

A = tokenize_inputs(df_train.answer.values, tokenizer,224, cut_class=False)

tT = tokenize_inputs(df_test.question_title.values, tokenizer, 64)

tQ = tokenize_inputs(df_test.question_body.values, tokenizer, 224)

tA = tokenize_inputs(df_test.answer.values, tokenizer, 224, cut_class=False)
X_train = np.concatenate([T, Q, A], axis=1)

X_test = np.concatenate([tT, tQ, tA], axis=1)
y_train = df_train[target_cols].values
def create_attn_masks(input_ids):

    """

    Create attention masks to tell model whether attention should be applied to

    the input id tokens. Do not want to perform attention on padding tokens.

    original author: Josh Xin Jie Lee @ Medium

    """

    # Create attention masks

    attention_masks = []



    # Create a mask of 1s for each token followed by 0s for padding

    for seq in input_ids:

        seq_mask = [float(i>0) for i in seq]

        attention_masks.append(seq_mask)

    return attention_masks

    

X_train_masks = create_attn_masks(X_train)

X_test_masks = create_attn_masks(X_test)
X_train, X_val, X_train_masks, X_val_masks, y_train, y_val = train_test_split(X_train, X_train_masks, y_train,

                                                                              test_size=0.15, random_state=46)
X_train = torch.tensor(X_train)

X_val = torch.tensor(X_val)



y_train = torch.tensor(y_train, dtype=torch.float32)

y_val = torch.tensor(y_val, dtype=torch.float32)



X_train_masks = torch.tensor(X_train_masks, dtype=torch.long)

X_val_masks = torch.tensor(X_val_masks, dtype=torch.long)
# Select a batch size for training

batch_size = 8



# Create an iterator of our data with torch DataLoader. This helps save on 

# memory during training because, unlike a for loop, 

# with an iterator the entire dataset does not need to be loaded into memory



train_data = TensorDataset(X_train, X_train_masks, y_train)

train_sampler = RandomSampler(train_data)

train_dataloader = DataLoader(train_data,\

                              sampler=train_sampler,\

                              batch_size=batch_size)



validation_data = TensorDataset(X_val, X_val_masks, y_val)

validation_sampler = SequentialSampler(validation_data)

validation_dataloader = DataLoader(validation_data,\

                                   sampler=validation_sampler,\

                                   batch_size=batch_size)
class XLNetForMultiLabelSequenceClassification(torch.nn.Module):

  

  def __init__(self, num_labels=2):

    super(XLNetForMultiLabelSequenceClassification, self).__init__()

    self.num_labels = num_labels

    self.xlnet = XLNetModel.from_pretrained('./xlnet-base-cased')

    self.classifier = torch.nn.Linear(768, num_labels)

    self.loss_fct = BCEWithLogitsLoss()

    self.dropout = torch.nn.Dropout(0.2)

    self.activation = torch.nn.ELU()

    

#     self.dense = nn.Linear(config.hidden_size, config.hidden_size)

#         self.activation = nn.Tanh()



    torch.nn.init.xavier_normal_(self.classifier.weight)



  def forward(self, input_ids, token_type_ids=None,\

              attention_mask=None, labels=None):

    # last hidden layer

    last_hidden_state = self.xlnet(input_ids=input_ids,\

                                   attention_mask=attention_mask,\

                                   token_type_ids=token_type_ids)

    # pool the outputs into a mean vector

    mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)

    mean_last_hidden_state = self.activation(mean_last_hidden_state)

    mean_last_hidden_state = self.dropout(mean_last_hidden_state)

    logits = self.classifier(mean_last_hidden_state)

        

    if labels is not None:

      loss = self.loss_fct(logits.view(-1, self.num_labels),\

                      labels.view(-1, self.num_labels))

      return loss

    else:

      return logits

    

  def freeze_xlnet_decoder(self):

    """

    Freeze XLNet weight parameters. They will not be updated during training.

    """

    for param in self.xlnet.parameters():

      param.requires_grad = False

    

  def unfreeze_xlnet_decoder(self):

    """

    Unfreeze XLNet weight parameters. They will be updated during training.

    """

    for param in self.xlnet.parameters():

      param.requires_grad = True

    

  def pool_hidden_state(self, last_hidden_state):

    """

    Pool the output vectors into a single mean vector 

    """

    last_hidden_state = last_hidden_state[0]

    mean_last_hidden_state = torch.mean(last_hidden_state[-128:], 1)

#     mean_last_hidden_state = last_hidden_state[-1, :]

    return mean_last_hidden_state



# len(Y_train[0]) = 6

model = XLNetForMultiLabelSequenceClassification(num_labels=len(y_train[0]))

# model.freeze_xlnet_decoder()
# optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01, correct_bias=False)

from scipy.stats import spearmanr

rho_bar = []

def train(model, num_epochs,\

#           optimizer,\

          train_dataloader, valid_dataloader,\

          model_save_path,\

          train_loss_set=[], valid_loss_set = [],\

          lowest_eval_loss=None, start_epoch=0,\

          device="cpu"

          ):

    """

    Train the model and save the model with the lowest validation loss

    """

    crit_function = nn.BCEWithLogitsLoss()

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=3.5e-5, weight_decay=0.01, correct_bias=False)

#     optimizer = torch.optim.Adamax(model.parameters(), lr=3e-5)

    # trange is a tqdm wrapper around the normal python range

    for i in trange(num_epochs, desc="Epoch"):

        # if continue training from saved model

        actual_epoch = start_epoch + i



        # Training



        # Set our model to training mode (as opposed to evaluation mode)

        model.train()



        # Tracking variables

        tr_loss = 0

        num_train_samples = 0



        t = tqdm(total=len(train_data), desc="Training: ", position=0)

        # Train the data for one epoch

        for step, batch in enumerate(train_dataloader):

            # Add batch to GPU

            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader

            b_input_ids, b_input_mask, b_labels = batch

            # Clear out the gradients (by default they accumulate)

            optimizer.zero_grad()

            # Forward pass

            loss = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

            # store train loss

            tr_loss += loss.item()

            num_train_samples += b_labels.size(0)

            # Backward pass

            loss.backward()

            # Update parameters and take a step using the computed gradient

            optimizer.step()

            #scheduler.step()

            t.update(n=b_input_ids.shape[0])

        t.close()

        # Update tracking variables

        epoch_train_loss = tr_loss/num_train_samples

        train_loss_set.append(epoch_train_loss)



        print("Train loss: {}".format(epoch_train_loss))



        # Validation



        # Put model in evaluation mode to evaluate loss on the validation set

        model.eval()



        # Tracking variables 

        eval_loss = 0

        num_eval_samples = 0



        v_preds = []

        v_labels = []



        # Evaluate data for one epoch

        t = tqdm(total=len(validation_data), desc="Validating: ", position=0)

        for batch in valid_dataloader:

            # Add batch to GPU

            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader

            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients,

            # saving memory and speeding up validation

            with torch.no_grad():

                # Forward pass, calculate validation loss

                preds = model(b_input_ids, attention_mask=b_input_mask)

                loss = crit_function(preds, b_labels)

                v_labels.append(b_labels.cpu().numpy())

                v_preds.append(preds.cpu().numpy())

                # store valid loss

                eval_loss += loss.item()

                num_eval_samples += b_labels.size(0)

            t.update(n=b_labels.shape[0])

        t.close()



        v_labels = np.vstack(v_labels)

        v_preds = np.vstack(v_preds)

        print(v_labels.shape)

        print(v_preds.shape)

        rho_val = np.mean([spearmanr(v_labels[:, ind] + np.random.normal(0, 1e-7, v_preds.shape[0]),

                                            v_preds[:, ind] + np.random.normal(0, 1e-7, v_preds.shape[0])).correlation for ind in range(v_preds.shape[1])]

                                )

        rho_bar.append([spearmanr(v_labels[:, ind] + np.random.normal(0, 1e-7, v_preds.shape[0]),

                                            v_preds[:, ind] + np.random.normal(0, 1e-7, v_preds.shape[0])).correlation for ind in range(v_preds.shape[1])])

        epoch_eval_loss = eval_loss/num_eval_samples

        valid_loss_set.append(epoch_eval_loss)



        print("Epoch #{}, training BCE loss: {}, validation BCE loss: ~{}, validation spearmanr: {}"\

                .format(0, epoch_train_loss, epoch_eval_loss, rho_val))



        if lowest_eval_loss == None:

            lowest_eval_loss = epoch_eval_loss

            # save model

        #   save_model(model, model_save_path, actual_epoch,\

        #              lowest_eval_loss, train_loss_set, valid_loss_set)

        else:

            if epoch_eval_loss < lowest_eval_loss:

                lowest_eval_loss = epoch_eval_loss

            # save model

            # save_model(model, model_save_path, actual_epoch,\

            #            lowest_eval_loss, train_loss_set, valid_loss_set)

        print("\n")



    return model, train_loss_set, valid_loss_set





def save_model(model, save_path, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist):

  """

  Save the model to the path directory provided

  Not used here!

  """

  model_to_save = model.module if hasattr(model, 'module') else model

  checkpoint = {'epochs': epochs, \

                'lowest_eval_loss': lowest_eval_loss,\

                'state_dict': model_to_save.state_dict(),\

                'train_loss_hist': train_loss_hist,\

                'valid_loss_hist': valid_loss_hist

               }

  torch.save(checkpoint, save_path)

  print("Saving model at epoch {} with validation loss of {}".format(epochs,\

                                                                     lowest_eval_loss))

  return

  

def load_model(save_path):

  """

  Load the model from the path directory provided

  Not used here!

  """

  checkpoint = torch.load(save_path)

  model_state_dict = checkpoint['state_dict']

  model = XLNetForMultiLabelSequenceClassification(num_labels=model_state_dict["classifier.weight"].size()[0])

  model.load_state_dict(model_state_dict)



  epochs = checkpoint["epochs"]

  lowest_eval_loss = checkpoint["lowest_eval_loss"]

  train_loss_hist = checkpoint["train_loss_hist"]

  valid_loss_hist = checkpoint["valid_loss_hist"]

  

  return model, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist
cwd = os.getcwd()

model_save_path = "./"

model, train_loss_set, valid_loss_set = train(model=model,\

                                              num_epochs = 4,

#                                               optimizer = optimizer,

                                              train_dataloader = train_dataloader,

                                              valid_dataloader = validation_dataloader,

                                              model_save_path = model_save_path,

                                              device='cuda'

                                              )
type(X_test)
rho_bar

fig = plt.figure()

plt.figure(figsize=(100,10))

fig.tight_layout()

plt.bar(x = target_cols, height = np.mean(np.array(rho_bar), 0))

plt.show()
test_data = TensorDataset(torch.tensor(X_test), torch.tensor(X_test_masks, dtype=torch.long))

test_dataloader = DataLoader(test_data,

                                   shuffle=False,

                                   batch_size=batch_size)


def generate_predictions(model, dataloader, num_labels, device="cpu", batch_size=8):



    pred_probs = np.array([]).reshape(0, num_labels)



    model.to(device)

    model.eval()



    for X, masks in dataloader:

        X = X.to(device)

        masks = masks.to(device)

        with torch.no_grad():

          logits = model(input_ids=X, attention_mask=masks)

          logits = logits.sigmoid().detach().cpu().numpy()

          pred_probs = np.vstack([pred_probs, logits])

    return pred_probs

num_labels = len(target_cols)

pred_probs = generate_predictions(model, test_dataloader, num_labels=30, device="cuda", batch_size=8)
df_submit[target_cols] = pred_probs

df_submit.question_not_really_a_question = np.random.normal(0, 1e-10, df_submit.question_not_really_a_question.values.shape)

df_submit.question_type_spelling = np.random.normal(0, 1e-10, df_submit.question_type_spelling.values.shape)

# Fix these value because in EDA, we saw these were almost all 0s.
df_submit.to_csv("submission.csv", index = False)

df_submit
del model

torch.cuda.empty_cache()