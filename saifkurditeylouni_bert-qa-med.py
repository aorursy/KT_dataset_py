!pip install transformers

!pip install wget



import torch

from transformers import BertForQuestionAnswering, AdamW, BertConfig, AutoTokenizer, AutoModelForQuestionAnswering

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split

from transformers import BertTokenizer

from transformers import get_linear_schedule_with_warmup

import wget

import os

import pandas as pd

import json

import numpy as np

import datetime

import time

import random

import gc

import tensorflow as tf

import pickle

import matplotlib.pyplot as plt

import seaborn as sns



# Function to calculate the accuracy of our predictions vs labels

def flat_accuracy(preds, labels):

    pred_flat = np.argmax(preds, axis=1).flatten()

    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)



def format_time(elapsed):

    '''

    Takes a time in seconds and returns a string hh:mm:ss

    '''

    # Round to the nearest second.

    elapsed_rounded = int(round((elapsed)))

    

    # Format as hh:mm:ss

    return str(datetime.timedelta(seconds=elapsed_rounded))



def extract(d):



  df=pd.DataFrame(columns=['sentence', 'answer_start', 'answer_end', 'question'])

  for index, row in d.iterrows():

    tmp=pd.DataFrame(columns=['sentence', 'answer_start', 'answer_end', 'question'])

    #tmp['sentence']=pd.Series([row['data']['story'] for a, q in zip(row['data']['answers'], row['data']['questions'])])

    #tmp['answer_start']=pd.Series([a['span_start']+len(q['input_text']) for a, q in zip(row['data']['answers'], row['data']['questions'])]).astype(str).astype(int)

    #tmp['answer_end']=pd.Series([a['span_end']+len(q['input_text']) for a, q in zip(row['data']['answers'], row['data']['questions'])]).astype(str).astype(int)

    tmp['sentence']=pd.Series([row['data']['story'][a['span_start']:] for a, q in zip(row['data']['answers'], row['data']['questions'])])

    tmp['answer_start']=pd.Series([len(q['input_text']) for a, q in zip(row['data']['answers'], row['data']['questions'])]).astype(str).astype(int)

    tmp['answer_end']=pd.Series([a['span_end']-a['span_start']+len(q['input_text']) for a, q in zip(row['data']['answers'], row['data']['questions'])]).astype(str).astype(int)

    tmp['question']=pd.Series([q['input_text'] for q in row['data']['questions']])

    df=df.append(tmp, ignore_index=True)

  return df



def getdevice():



    # If there's a GPU available...

    if torch.cuda.is_available():    



        # Tell PyTorch to use the GPU.    

        device = torch.device("cuda")



        print('There are %d GPU(s) available.' % torch.cuda.device_count())



        print('We will use the GPU:', torch.cuda.get_device_name(0))



    # If not...

    else:

        print('No GPU available, using the CPU instead.')

        device = torch.device("cpu")

    return device



def preprocessing():





    print('Downloading dataset...')



    # The URL for the dataset zip file.

    url = 'http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json'



    # Download the file (if we haven't already)

    if not os.path.exists('./coqa-train-v1.0.json'):

        wget.download(url, './coqa-train-v1.0.json')



    with open('./coqa-train-v1.0.json') as json_file:

        data = pd.read_json(json_file)

        del data['version']

        data=data['data'].to_frame()



    data=extract(data).reset_index(drop=True)

    

    # Report the number of sentences.

    print('Number of training sentences: {:,}\n'.format(data.shape[0]))





    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')#BertTokenizer



    #model = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')



    #question = "How many parameters does BERT-large have?"

    #answer_text = "BERT-large is really big... it has 24-layers and an embedding size of 1,024, for a total of 340M parameters! Altogether it is 1.34GB, so expect it to take a couple minutes to download to your Colab instance."

    question="How old are you?"

    answer_text="Africa is a continent, as is Asia. It has been said by the ancients, and their elders before them..., that I was possibly twenty seven if I was a day"



    input_ids_list=[]

    tokens_list=[]

    segment_ids_list=[]

    attention_masks_list=[]

    answerstart=[]

    answerend=[]

    for index, row in data.iterrows():

      # Apply the tokenizer to the input text, treating them as a text-pair.

      encoded_dict = tokenizer.encode_plus(row['question'], row['sentence'], max_length = 128, truncation=True,# Pad & truncate all sentences.

                            pad_to_max_length = True,

                            return_attention_mask = True,  # Construct attn. masks.

                            return_tensors = 'pt',)

      input_ids = encoded_dict['input_ids']

      input_ids_list.append(input_ids.numpy())

      attention_masks_list.append(encoded_dict['attention_mask'].numpy())



      #print('The input has a total of {:} tokens.'.format(len(input_ids)))



      #BERT only needs the token IDs, but for the purpose of inspecting the 

      # tokenizer's behavior, let's also get the token strings and display them.

      #tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

      #tokens_list.append(tokens)



      # For each token and its id...

      """for token, id in zip(tokens, input_ids):



          # If this is the [SEP] token, add some space around it to make it stand out.

          if id == tokenizer.sep_token_id:

              print('')



          # Print the token string and its ID in two columns.

          print('{:<12} {:>6,}'.format(token, id))



          if id == tokenizer.sep_token_id:

              print('')"""





      # Search the input_ids for the first instance of the `[SEP]` token.

      sep_index = input_ids[0].tolist().index(tokenizer.sep_token_id)

      # The number of segment A tokens includes the [SEP] token istelf.

      num_seg_a = sep_index + 1

      # The remainder are segment B.

      num_seg_b = len(input_ids[0]) - num_seg_a

      # Construct the list of 0s and 1s.

      segment_ids = [0]*num_seg_a + [1]*num_seg_b



      # There should be a segment_id for every input token.

      assert len(segment_ids) == len(input_ids[0])

      segment_ids_list.append(segment_ids)

      

      answerstart.append(row['answer_start']+num_seg_a)

      answerend.append(row['answer_end']+num_seg_a)

        



    print("done tokenizing\n")



    

    df=pd.DataFrame()

    df['input_ids'] = pd.Series(input_ids_list)

    df['attention_masks'] = pd.Series(attention_masks_list)

    df['segment_ids'] = pd.Series(segment_ids_list)

    df['start_labels'] = pd.Series(answerstart)#pd.Series(data['answer_start'].values.tolist())  

    df['end_labels'] = pd.Series(answerend)#pd.Series(data['answer_end'].values.tolist())

    

    df=df[df['end_labels']<128]



    del input_ids_list

    del attention_masks_list

    del segment_ids_list

    del data



    

    print(df.head())

    path='/kaggle/working/'

    tokenizer.save_pretrained(path)

    path=os.path.join(path, "data")

    df.to_csv(path, index=False)

    print('saved\n')



    

    del df

    gc.collect()

    

    

def processing():





    print('bueno')

    

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            if filename=='data_128c':

                table_path_input=os.path.join(dirname, filename)

    df=pd.read_csv(table_path_input)

        

    #.reshape(1,np.fromstring(t.strip(']['), dtype=np.int, sep=' ').shape[0])

    input_ids=[torch.from_numpy(np.fromstring(t.strip(']['), dtype=np.int, sep=' ').reshape(1,np.fromstring(t.strip(']['), dtype=np.int, sep=' ').shape[0])) for t in df['input_ids'].values.tolist()]

    input_ids = torch.cat(input_ids, dim=0)

    print(input_ids[0])

    attention_masks=[torch.from_numpy(np.fromstring(t.strip(']['), dtype=np.int, sep=' ').reshape(1,np.fromstring(t.strip(']['), dtype=np.int, sep=' ').shape[0])) for t in df['attention_masks'].values.tolist()]

    attention_masks = torch.cat(attention_masks, dim=0)

    print(attention_masks[0])#!!!!!!add reshape to have it 2D

    #torch.from_numpy(np.fromstring(t, dtype=np.int, sep=','))

    segment_ids=[torch.from_numpy(np.fromstring(t.strip(']['), dtype=np.int, sep=',').reshape(1,np.fromstring(t.strip(']['), dtype=np.int, sep=',').shape[0])) for t in df['segment_ids'].values.tolist()]

    segment_ids = torch.cat(segment_ids, dim=0)

    print(segment_ids[0])

    start_labels = torch.tensor(df['start_labels'].values)

    print(start_labels)

    end_labels = torch.tensor(df['end_labels'].values)



    # ==========================

    # Combine the training inputs into a TensorDataset.

    dataset = TensorDataset(input_ids, attention_masks, segment_ids, start_labels, end_labels)

    

    del input_ids

    del attention_masks

    del segment_ids

    del start_labels

    del end_labels

    del df

    



    # Create a 90-10 train-validation split.



    # Calculate the number of samples to include in each set.

    train_size = int(0.9 * len(dataset))

    val_size = len(dataset) - train_size



    # Divide the dataset by randomly selecting samples.

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    del dataset

    gc.collect()



    print('{:>5,} training samples'.format(train_size))

    print('{:>5,} validation samples'.format(val_size))



    # The DataLoader needs to know our batch size for training, so we specify it 

    # here. For fine-tuning BERT on a specific task, the authors recommend a batch 

    # size of 16 or 32.

    batch_size = 16



    # Create the DataLoaders for our training and validation sets.

    # We'll take training samples in random order. 

    train_dataloader = DataLoader(

                train_dataset,  # The training samples.

                sampler = RandomSampler(train_dataset), # Select batches randomly

                batch_size = batch_size # Trains with this batch size.

            )



    # For validation the order doesn't matter, so we'll just read them sequentially.

    validation_dataloader = DataLoader(

                val_dataset, # The validation samples.

                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.

                batch_size = batch_size # Evaluate with this batch size.

            )

    

    del train_dataset

    del val_dataset

    gc.collect()

    

    # Load BertForQuestionAnswering, the pretrained BERT model with a single 

    # linear classification layer on top. 

    model = BertForQuestionAnswering.from_pretrained(

        "bert-large-uncased-whole-word-masking-finetuned-squad", # Use the 12-layer BERT model, with an uncased vocab.   

        output_attentions = False, # Whether the model returns attentions weights.

        output_hidden_states = False, # Whether the model returns all hidden-states.

    )#AutoModelForQuestionAnswering



    # Tell pytorch to run this model on the GPU.

    #model.cuda()





    # Get all of the model's parameters as a list of tuples.

    params = list(model.named_parameters())



    """print('The BERT model has {:} different named parameters.\n'.format(len(params)))



    print('==== Embedding Layer ====\n')



    for p in params[0:5]:

        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))



    print('\n==== First Transformer ====\n')



    for p in params[5:21]:

        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))



    print('\n==== Output Layer ====\n')



    for p in params[-4:]:

        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))"""





    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 

    # I believe the 'W' stands for 'Weight Decay fix"

    optimizer = AdamW(model.parameters(),

                      lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5

                      eps = 1e-8 # args.adam_epsilon  - default is 1e-8.

                    )



    # Number of training epochs. The BERT authors recommend between 2 and 4. 

    # We chose to run for 4, but we'll see later that this may be over-fitting the

    # training data.

    epochs = 4



    # Total number of training steps is [number of batches] x [number of epochs]. 

    # (Note that this is not the same as the number of training samples).

    total_steps = len(train_dataloader) * epochs



    # Create the learning rate scheduler.

    scheduler = get_linear_schedule_with_warmup(optimizer, 

                                                num_warmup_steps = 0, # Default value in run_glue.py

                                                num_training_steps = total_steps)





    # This training code is based on the `run_glue.py` script here:

    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128



    # Set the seed value all over the place to make this reproducible.

    seed_val = 42



    random.seed(seed_val)

    np.random.seed(seed_val)

    torch.manual_seed(seed_val)

    torch.cuda.manual_seed_all(seed_val)



    # We'll store a number of quantities such as training and validation loss, 

    # validation accuracy, and timings.

    training_stats = []



    # Measure the total training time for the whole run.

    total_t0 = time.time()



    # For each epoch...

    for epoch_i in range(0, epochs):



        # ========================================

        #               Training

        # ========================================



        # Perform one full pass over the training set.



        print("")

        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

        print('Training...')



        # Measure how long the training epoch takes.

        t0 = time.time()



        # Reset the total loss for this epoch.

        total_train_loss = 0



        # Put the model into training mode. Don't be mislead--the call to 

        # `train` just changes the *mode*, it doesn't *perform* the training.

        # `dropout` and `batchnorm` layers behave differently during training

        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)

        model.train()



        # For each batch of training data...

        for step, batch in enumerate(train_dataloader):

            

            # Progress update every 40 batches.

            if step % 40 == 0 and not step == 0:

                # Calculate elapsed time in minutes.

                elapsed = format_time(time.time() - t0)



                # Report progress.

                #print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))



            # Unpack this training batch from our dataloader. 

            #

            # As we unpack the batch, we'll also copy each tensor to the GPU using the 

            # `to` method.

            #

            # `batch` contains three pytorch tensors:

            #   [0]: input ids 

            #   [1]: attention masks

            #   [2]: labels 

            b_input_ids = batch[0].to(device)

            b_input_mask = batch[1].to(device)

            b_input_segment = batch[2].to(device)

            start_labels = batch[3].to(device)#torch.tensor([5])#, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])#batch[3].to(device)

            end_labels = batch[4].to(device)#torch.tensor([10])#, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])#batch[4].to(device)

            

            print(start_labels)

            print(end_labels)

            print(b_input_segment)

            print(b_input_mask)

            print(b_input_ids)

            

            del batch

            gc.collect()



            # Always clear any previously calculated gradients before performing a

            # backward pass. PyTorch doesn't do this automatically because 

            # accumulating the gradients is "convenient while training RNNs". 

            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)

            model.zero_grad()        



            # Perform a forward pass (evaluate the model on this training batch).

            # The documentation for this `model` function is here: 

            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification

            # It returns different numbers of parameters depending on what arguments

            # arge given and what flags are set. For our useage here, it returns

            # the loss (because we provided labels) and the "logits"--the model

            # outputs prior to activation.

            model.to(device)

            loss, start_scores, end_scores = model(b_input_ids, 

                                             token_type_ids=b_input_segment, 

                                             attention_mask=b_input_mask, 

                                             start_positions=start_labels,

                                             end_positions=end_labels)

            



            # Accumulate the training loss over all of the batches so that we can

            # calculate the average loss at the end. `loss` is a Tensor containing a

            # single value; the `.item()` function just returns the Python value 

            # from the tensor.

            total_train_loss += loss.item()

            



            # Perform a backward pass to calculate the gradients.

            loss.backward()

            

            del start_labels

            del end_labels

            del b_input_segment

            del b_input_mask

            del b_input_ids

            del loss

            del start_scores

            del end_scores

            gc.collect()



            # Clip the norm of the gradients to 1.0.

            # This is to help prevent the "exploding gradients" problem.

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)



            # Update parameters and take a step using the computed gradient.

            # The optimizer dictates the "update rule"--how the parameters are

            # modified based on their gradients, the learning rate, etc.

            optimizer.step()



            # Update the learning rate.

            scheduler.step()



        # Calculate the average loss over all of the batches.

        avg_train_loss = total_train_loss / len(train_dataloader)            



        # Measure how long this epoch took.

        training_time = format_time(time.time() - t0)



        print("")

        print("  Average training loss: {0:.2f}".format(avg_train_loss))

        print("  Training epcoh took: {:}".format(training_time))



        # ========================================

        #               Validation

        # ========================================

        # After the completion of each training epoch, measure our performance on

        # our validation set.



        print("")

        print("Running Validation...")



        t0 = time.time()



        # Put the model in evaluation mode--the dropout layers behave differently

        # during evaluation.

        model.eval()



        # Tracking variables 

        total_eval_accuracy = 0

        total_eval_loss = 0

        nb_eval_steps = 0



        # Evaluate data for one epoch

        for batch in validation_dataloader:



            # Unpack this training batch from our dataloader. 

            #

            # As we unpack the batch, we'll also copy each tensor to the GPU using 

            # the `to` method.

            #

            # `batch` contains three pytorch tensors:

            #   [0]: input ids 

            #   [1]: attention masks

            #   [2]: labels 

            b_input_ids = batch[0].to(device)

            b_input_mask = batch[1].to(device)

            b_input_segment = batch[2].to(device)

            start_labels = batch[3].to(device)

            end_labels = batch[4].to(device)





            # Tell pytorch not to bother with constructing the compute graph during

            # the forward pass, since this is only needed for backprop (training).

            with torch.no_grad():        



                # Forward pass, calculate logit predictions.

                # token_type_ids is the same as the "segment ids", which 

                # differentiates sentence 1 and 2 in 2-sentence tasks.

                # The documentation for this `model` function is here: 

                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification

                # Get the "logits" output by the model. The "logits" are the output

                # values prior to applying an activation function like the softmax.

                model.to(device)

                loss, start_scores, end_scores = model(b_input_ids, 

                                                 token_type_ids=b_input_segment, 

                                                 attention_mask=b_input_mask, 

                                                 start_positions=start_labels,

                                                 end_positions=end_labels)



            # Accumulate the validation loss.

            total_eval_loss += loss.item()



            # Move logits and labels to CPU

            start_scores = start_scores.detach().cpu().numpy()

            end_scores = end_scores.detach().cpu().numpy()

            start_labels = start_labels.to('cpu').numpy()

            end_labels = end_labels.to('cpu').numpy()



            # Calculate the accuracy for this batch of test sentences, and

            # accumulate it over all batches.

            total_eval_accuracy += (flat_accuracy(start_scores, start_labels)+flat_accuracy(end_scores, end_labels))/2

            

            del start_labels

            del end_labels

            del b_input_segment

            del b_input_mask

            del b_input_ids

            del loss

            del start_scores

            del end_scores

            gc.collect()





        # Report the final accuracy for this validation run.

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)

        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))



        # Calculate the average loss over all of the batches.

        avg_val_loss = total_eval_loss / len(validation_dataloader)



        # Measure how long the validation run took.

        validation_time = format_time(time.time() - t0)



        print("  Validation Loss: {0:.2f}".format(avg_val_loss))

        print("  Validation took: {:}".format(validation_time))



        # Record all statistics from this epoch.

        training_stats.append(

            {

                'epoch': epoch_i + 1,

                'Training Loss': avg_train_loss,

                'Valid. Loss': avg_val_loss,

                'Valid. Accur.': avg_val_accuracy,

                'Training Time': training_time,

                'Validation Time': validation_time

            }

        )



    print("")

    print("Training complete!")



    path='/kaggle/working/'

    model.save_pretrained(path)

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    

    

def find_answer():

    # ======== Evaluate ========



    question="What is the name of the cow?"

    answer_text="The cow has a name, it is Lassie"



    path='/kaggle/input/model-cola-64c'

    model = BertForQuestionAnswering.from_pretrained(path)

    

    tokenizer = BertTokenizer.from_pretrained(path)#'bert-large-uncased-whole-word-masking-finetuned-squad'



    model.eval()



    # Run our example through the model.

    encoded_dict = tokenizer.encode_plus(question, answer_text, max_length = 128, truncation=True,# Pad & truncate all sentences.

                            pad_to_max_length = True,

                            return_attention_mask = True,  # Construct attn. masks.

                            return_tensors = 'pt',)



    # Search the input_ids for the first instance of the `[SEP]` token.

    sep_index = encoded_dict['input_ids'][0].tolist().index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.

    num_seg_a = sep_index + 1

    # The remainder are segment B.

    num_seg_b = len(encoded_dict['input_ids'][0]) - num_seg_a

    # Construct the list of 0s and 1s.

    segment_ids = [0]*num_seg_a + [1]*num_seg_b



    # There should be a segment_id for every input token.

    assert len(segment_ids) == len(encoded_dict['input_ids'][0].tolist())

    

    # Get the string versions of the input tokens.

    tokens = tokenizer.convert_ids_to_tokens(encoded_dict['input_ids'][0].tolist())

    



    start_scores, end_scores = model(encoded_dict['input_ids'], # The tokens representing our input text.

                                     attention_mask=encoded_dict['attention_mask'],

                                     token_type_ids=torch.tensor(segment_ids)) # The segment IDs to differentiate question from answer_text



    preds_mask = (

    (encoded_dict['input_ids'] != tokenizer.vocab["[SEP]"])

    & (encoded_dict['input_ids'] != tokenizer.vocab["[PAD]"])

    & (encoded_dict['input_ids'] != tokenizer.vocab["[CLS]"])

    )

    



    # ======== Reconstruct Answer ========



    # Find the tokens with the highest `start` and `end` scores.

    answer_start = torch.argmax(start_scores[preds_mask])

    answer_end = torch.argmax(end_scores[preds_mask])



    # Start with the first token.

    answer = tokens[answer_start]



    # Select the remaining answer tokens and join them with whitespace.

    for i in range(answer_start + 1, answer_end + 1):



        # If it's a subword token, then recombine it with the previous token.

        if tokens[i][0:2] == '##':

            answer += tokens[i][2:]



        # Otherwise, add a space then the token.

        else:

            answer += ' ' + tokens[i]

        

    print('Answer: "' + answer + '"')





    # Use plot styling from seaborn.

    sns.set(style='darkgrid')



    # Increase the plot size and font size.

    #sns.set(font_scale=1.5)

    plt.rcParams["figure.figsize"] = (16,8)

            

    # Pull the scores out of PyTorch Tensors and convert them to 1D numpy arrays.

    s_scores = start_scores.detach().numpy().flatten()

    e_scores = end_scores.detach().numpy().flatten()

    

    print(s_scores)





    # We'll use the tokens as the x-axis labels. In order to do that, they all need

    # to be unique, so we'll add the token index to the end of each one.

    token_labels = []

    for (i, token) in enumerate(tokens):

        token_labels.append('{:} - {:>2}'.format(token, i))



    # Create a barplot showing the start word score for all of the tokens.

    ax = sns.barplot(x=token_labels, y=s_scores, ci=None)



    # Turn the xlabels vertical.

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")



    # Turn on the vertical grid to help align words to scores.

    ax.grid(True)



    plt.title('Start Word Scores')



    plt.show()

    

    # Create a barplot showing the end word score for all of the tokens.

    ax = sns.barplot(x=token_labels, y=e_scores, ci=None)



    # Turn the xlabels vertical.

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")



    # Turn on the vertical grid to help align words to scores.

    ax.grid(True)



    plt.title('End Word Scores')



    plt.show()



device=getdevice()

preprocessing()    

#processing()

#find_answer()

 