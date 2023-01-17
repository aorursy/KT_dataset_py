import pandas as pd

import numpy as np

from tqdm import tqdm, trange



data = pd.read_csv("../input/entity-annotated-corpus/ner_dataset.csv", 

                   encoding="latin1").fillna(method="ffill")

data.tail().T
class GetSentence(object):



    def __init__(self, data):

        self.n_sent = 1

        self.data = data

        self.empty = False

        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),

                                                           s["POS"].values.tolist(),

                                                           s["Tag"].values.tolist())]

        self.grouped = self.data.groupby("Sentence #").apply(agg_func)

        self.sentences = [s for s in self.grouped]



    def get_next(self):

        try:

            s = self.grouped["Sentence: {}".format(self.n_sent)]

            self.n_sent += 1

            return s

        except:

            return None
getter = GetSentence(data)
sentences = [[word[0] for word in sentence] for sentence in getter.sentences]

sentences[0]
labels = [[s[2] for s in sentence] for sentence in getter.sentences]

labels[0]
tag_values = list(set(data["Tag"].values))

tag_values.append("PAD")

tag2idx = {t: i for i, t in enumerate(tag_values)}



tag2idx
import torch

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, BertConfig



from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split



torch.__version__
MAX_LEN = 75

bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_gpu = torch.cuda.device_count()



torch.cuda.get_device_name(0)
tokenizer = BertTokenizer.from_pretrained('/kaggle/input/scibert-scivocab-uncased/scibert_scivocab_uncased',

                                          do_lower_case=False)
def tokenize_and_preserve_labels(sentence, text_labels):

    tokenized_sentence = []

    labels = []



    for word, label in zip(sentence, text_labels):



        # Tokenize the word and count number of subwords the word is broken into

        tokenized_word = tokenizer.tokenize(word)

        n_subwords = len(tokenized_word)



        # Add the tokenized word to the final tokenized word list

        tokenized_sentence.extend(tokenized_word)



        # Add the same label to the new list of labels `n_subwords` times

        labels.extend([label] * n_subwords)



    return tokenized_sentence, labels
tokenized_texts_and_labels = [

    tokenize_and_preserve_labels(sent, labs)

    for sent, labs in zip(sentences, labels)

]
tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]



labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],

                          maxlen=MAX_LEN,

                          dtype="long",

                          value=0.0,

                          truncating="post",

                          padding="post")
tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],

                     maxlen=MAX_LEN, 

                     value=tag2idx["PAD"],

                     padding="post",

                     dtype="long", 

                     truncating="post")
attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]
tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, 

                                                            tags,

                                                            random_state=0,

                                                            test_size=0.1)

tr_masks, val_masks, _, _ = train_test_split(attention_masks,

                                             input_ids,

                                             random_state=0,

                                             test_size=0.1)
tr_inputs = torch.tensor(tr_inputs)

val_inputs = torch.tensor(val_inputs)

tr_tags = torch.tensor(tr_tags)

val_tags = torch.tensor(val_tags)

tr_masks = torch.tensor(tr_masks)

val_masks = torch.tensor(val_masks)
train_data = TensorDataset(tr_inputs,

                           tr_masks,

                           tr_tags)

train_sampler = RandomSampler(train_data)

train_dataloader = DataLoader(train_data,

                              sampler=train_sampler, 

                              batch_size=bs)



valid_data = TensorDataset(val_inputs,

                           val_masks, 

                           val_tags)

valid_sampler = SequentialSampler(valid_data)

valid_dataloader = DataLoader(valid_data,

                              sampler=valid_sampler,

                              batch_size=bs)
import transformers

from transformers import BertForTokenClassification, AdamW
model = BertForTokenClassification.from_pretrained(

    '/kaggle/input/scibert-scivocab-uncased/scibert_scivocab_uncased',

    num_labels=len(tag2idx),

    output_attentions = False,

    output_hidden_states = False

)
model.cuda()



FULL_FINETUNING = True

if FULL_FINETUNING:

    

    param_optimizer = list(model.named_parameters())

    

    no_decay = ['bias', 'gamma', 'beta']

    

    optimizer_grouped_parameters = [

        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],

         'weight_decay_rate': 0.01},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],

         'weight_decay_rate': 0.0}

    ]

else:

    param_optimizer = list(model.classifier.named_parameters())

    

    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]



optimizer = AdamW(

    optimizer_grouped_parameters,

    lr=4e-5,

    eps=1e-8

)

from transformers import get_linear_schedule_with_warmup



epochs = 3

max_grad_norm = 1.0



# Total number of training steps is number of batches * number of epochs.

total_steps = len(train_dataloader) * epochs



# Create the learning rate scheduler.

scheduler = get_linear_schedule_with_warmup(

    optimizer,

    num_warmup_steps=0,

    num_training_steps=total_steps

)
!pip install seqeval

from seqeval.metrics import f1_score



def flat_accuracy(preds, labels):

    pred_flat = np.argmax(preds, axis=2).flatten()

    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Store the average loss after each epoch so we can plot them.

loss_values, validation_loss_values = [], []



for _ in trange(epochs, desc="Epoch "):

    # ========================================

    #               Training

    # ========================================

    # Perform one full pass over the training set.



    # Put the model into training mode.

    model.train()

    # Reset the total loss for this epoch.

    total_loss = 0



    # Training loop

    for step, batch in enumerate(train_dataloader):

        

        # add batch to gpu

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        

        # Always clear any previously calculated gradients before performing a backward pass.

        model.zero_grad()

        

        # forward pass

        # This will return the loss (rather than the model output)

        # because we have provided the `labels`.

        outputs = model(b_input_ids, 

                        token_type_ids=None,

                        attention_mask=b_input_mask, 

                        labels=b_labels)

        

        # get the loss

        loss = outputs[0]

        

        # Perform a backward pass to calculate the gradients.

        loss.backward()

        

        # track train loss

        total_loss += loss.item()

        

        # Clip the norm of the gradient

        # This is to help prevent the "exploding gradients" problem.

        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), 

                                       max_norm=max_grad_norm)

        

        # update parameters

        optimizer.step()

        

        # Update the learning rate.

        scheduler.step()



    # Calculate the average loss over the training data.

    avg_train_loss = total_loss / len(train_dataloader)

    print(f'Average train loss: {avg_train_loss}')



    # Store the loss value for plotting the learning curve.

    loss_values.append(avg_train_loss)





    # ========================================

    #               Validation

    # ========================================

    # After the completion of each training epoch, measure our performance on

    # our validation set.



    # Put the model into evaluation mode

    model.eval()

    

    # Reset the validation loss for this epoch.

    eval_loss, eval_accuracy = 0, 0

    

    nb_eval_steps, nb_eval_examples = 0, 0

    predictions , true_labels = [], []

    

    for batch in (valid_dataloader):

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch



        # Telling the model not to compute or store gradients,

        # saving memory and speeding up validation

        with torch.no_grad():

            

            # Forward pass, calculate logit predictions.

            # This will return the logits rather than the loss because we have not provided labels.

            outputs = model(b_input_ids, 

                            token_type_ids=None,

                            attention_mask=b_input_mask, 

                            labels=b_labels)

        

        # Move logits and labels to CPU

        logits = outputs[1].detach().cpu().numpy()

        label_ids = b_labels.to('cpu').numpy()



        # Calculate the accuracy for this batch of test sentences.

        eval_loss += outputs[0].mean().item()

        eval_accuracy += flat_accuracy(logits, label_ids)

        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

        true_labels.extend(label_ids)



        nb_eval_examples += b_input_ids.size(0)

        nb_eval_steps += 1



    eval_loss = eval_loss / nb_eval_steps

    validation_loss_values.append(eval_loss)

    print(f'Validation loss: {eval_loss}')

    print(f'Validation Accuracy: {eval_accuracy / nb_eval_steps}')

    

    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)

                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]

    valid_tags = [tag_values[l_i] for l in true_labels

                                  for l_i in l if tag_values[l_i] != "PAD"]

    print(f'Validation F1-Score: {f1_score(pred_tags, valid_tags)}')
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid')



# Increase the plot size and font size

plt.rcParams["figure.figsize"] = (9, 4)



# Plot the learning curve.

plt.plot(loss_values, 'b-o', label="training loss")

plt.plot(validation_loss_values, 'r-o', label="validation loss")



# Label the plot.

plt.title("Learning curve")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()



plt.show()
test_sentence = """

Prime Minister Narendra Modi holds a detailed meeting to discuss financial sector, 

structural and welfare measures to spur growth in Maharashtra, Delhi, Karnataka, Tamilnadu 

"""
tokenized_sentence = tokenizer.encode(test_sentence)



input_ids = torch.tensor([tokenized_sentence]).cuda()



with torch.no_grad():

    output = model(input_ids)

label_indices = np.argmax(output[0].to('cpu').numpy(),

                          axis=2)



# join bpe split tokens

tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])

new_tokens, new_labels = [], []



for token, label_idx in zip(tokens, label_indices[0]):

    if token.startswith("##"):

        new_tokens[-1] = new_tokens[-1] + token[2:]

    else:

        new_labels.append(tag_values[label_idx])

        new_tokens.append(token)



for token, label in zip(new_tokens, new_labels):

    print(f'{label}  {token}')

        