# turn the internet on for this to install properly

!pip install transformers

!pip install colorama
import numpy as np

import torch

import pandas as pd

import transformers

from tqdm import tnrange

from transformers import AlbertTokenizer, AlbertForQuestionAnswering

import colorama

import os

import re

import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tnrange



def get_filtered_articles(path_to_filtered_articles, path_to_metadata):



  # Get filtered articles 

  data_filtered_titles = pd.read_csv(path_to_filtered_articles, usecols=['title'], keep_default_na = False) # length: 15291

  metadata = pd.read_csv(path_to_metadata, keep_default_na=False) # length: 138794



  # Get detailed information of the filtered articles from metadata

  df = pd.merge(data_filtered_titles, metadata) # length: 13667



  return df



import pandas as pd

from tqdm import tnrange





# the file path containing titles of the filtered articles extracted by topic modeling

path_to_filtered_articles = '../input/covid19-related-articles/Ten_Tasks.csv'

path_to_metadata = '../input/CORD-19-research-challenge/metadata.csv'



data = get_filtered_articles(path_to_filtered_articles, path_to_metadata)

print(data.head())
# Import fine-tuned ALBERT model. 



from transformers import AlbertTokenizer, AlbertForQuestionAnswering



def get_model(path_to_model):



  if torch.cuda.is_available():  

    print("GPU available")

    dev = "cuda:0" 

  else:  

    dev = "cpu"  



  print("Import pre-trained ALBERT model...")

  tokenizer = AlbertTokenizer.from_pretrained(path_to_model)

  model = AlbertForQuestionAnswering.from_pretrained(

      path_to_model).to(dev)

    

  print("Import Complete!")

  return tokenizer, model, dev
path_to_model = '../input/albert-trained-on-squad-and-bioasq'



tokenizer, model, device = get_model(path_to_model)
import torch

from tqdm import tnrange



def extract_relevant_answer(model, tokenizer, data, question, device):

  confidence = []

  predictions = []

  start_end_idx = []

    

  for i in tnrange(len(data)):

    abstract = data.iloc[i]['abstract']

    input_ids = tokenizer.encode(question, abstract)

    input_ids = input_ids[0:512]

    

    sep_idx = input_ids.index(tokenizer.sep_token_id)

    token_type_ids = [0 if i <= sep_idx else 1 for i in range(len(input_ids))]



    input_ids_tensor = torch.tensor([input_ids]).to(device)

    token_type_ids_tensor = torch.tensor([token_type_ids]).to(device)

    

    start_scores, end_scores = model(input_ids_tensor, token_type_ids=token_type_ids_tensor)



    start_idx = start_scores.argmax()

    end_idx = end_scores.argmax()+1

    score = (start_scores.max() + end_scores.max()) / 2

    score = score.item()



    if start_idx <= 0 or end_idx <= 0 or end_idx<=start_idx:

      predictions.append("Not Relevant")

      score = float('-inf')

    else:

      tokens = tokenizer.convert_ids_to_tokens(input_ids[start_idx:end_idx])

      prediction = tokenizer.convert_tokens_to_string(tokens)

      predictions.append(prediction)

    

    confidence.append(score)

    start_end_idx.append([start_idx, end_idx])

    

  return predictions, confidence, start_end_idx
import pandas as pd

import numpy as np



def top_n_answers(question, predictions, confidence, start_end_idx, data, n):



  print("Get top " + str(n) + " relevant articles & answers")

  confidence = np.array(confidence)

  top_n_scores_idx = (-confidence).argsort()[:n]



  top_n_articles = pd.DataFrame()

  for idx in top_n_scores_idx:

    entry = data.iloc[idx]

    entry['confidence'] = confidence[idx]

    entry['prediction'] = predictions[idx]

    top_n_articles = top_n_articles.append(entry)

    

  top_n_articles.to_csv(question + ".csv")



  return top_n_articles
import colorama



def print_top_n_articles(question, top_n_articles):



  print("Prediction highlighted in red....")

  print("========  " + question + "  ======== ")

    

  for i in range(len(top_n_articles)):

    entry = top_n_articles.iloc[i]



    abstract = entry['abstract']

    prediction = entry['prediction']



    prediction_start_idx = abstract.find(prediction)

    prediction_end_idx = prediction_start_idx + len(prediction)

    

    

    print( "(" + str(i) + ")")

    print("Title : " + entry['title'] + "\n")

    print("Confidence: " + str(entry['confidence']))

    print("Abstract : " + abstract[:prediction_start_idx])

    print (colorama.Fore.RED, abstract[prediction_start_idx: prediction_end_idx].rstrip())

    print(colorama.Style.RESET_ALL, abstract[prediction_end_idx:] + "\n")





question = 'What do we know about vaccines and therapeutics?'



file_path = os.path.join('../input/covid19-top-articles', question.replace('?','').replace(',','') + '_.csv')



if os.path.exists(file_path):

    top_n_articles = pd.read_csv(file_path)

else:

    # Extract relevant span of text

    predictions, confidence, start_end_idx = extract_relevant_answer(model, tokenizer, data, question, device)



    # Get the top n articles sorted by confidence scores

    n = 10

    top_n_articles = top_n_answers(question, predictions, confidence, start_end_idx, data, n)



# Print top n relevant articles

print_top_n_articles(question, top_n_articles)
# STEP 1: 

# the file path containing titles of the filtered articles extracted by topic modeling

path_to_filtered_articles = '../input/covid19-related-articles/Ten_Tasks.csv'

path_to_metadata = '../input/CORD-19-research-challenge/metadata.csv'

# Get filtered articles

data = get_filtered_articles(path_to_filtered_articles, path_to_metadata)



# STEP 2: Load the pretrained model

path_to_model = '../input/albert-trained-on-squad-and-bioasq'

tokenizer, model, device = get_model(path_to_model)



# STEP 3: Extract Excerpt From Abstract Using Fine-Tuned ALBERT

questions = ['What is known about transmission, incubation, and environmental stability?',

             'What do we know about COVID-19 risk factors?',

             'What do we know about vaccines and therapeutics?',

             'What do we know about virus genetics, origin, and evolution?',

             'What has been published about medical care?',

             'What do we know about non-pharmaceutical interventions?',

             'What has been published about ethical and social science considerations?',

             'What do we know about diagnostics and surveillance?',

             'What has been published about information sharing and inter-sectoral collaboration?',

             'What do we know about the virus, the human immune response and predictive models?']



confidence_matrix = np.zeros(shape=(10,10))

for i in range(len(questions)):

    question = questions[i]

    file_path = os.path.join('../input/covid19-top-articles', question.replace('?','').replace(',','') + '_.csv')

    print(file_path)

    if os.path.exists(file_path):

        top_n_articles = pd.read_csv(file_path)

    else:

        print("File doesn't exist!")

        predictions, confidence, start_end_idx = extract_relevant_answer(model, tokenizer, data, question, device)

        

        # STEP 4: Get the top n articles sorted by confidence scores

        n = 10

        top_n_articles = top_n_answers(question, predictions, confidence, start_end_idx, data, n)

        

    # Get confidence score and store in numpy array

    confidence_matrix[i] = top_n_articles['confidence']

    # Print top n relevant articles

    print_top_n_articles(question, top_n_articles)
# Lables on x axis

x = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th']



# Plot a bar chart for each question.

fig, axs = plt.subplots(6,2,figsize=(15,15))



for i in range(len(confidence_matrix)):

    row = int(i / 2)

    col = int(i % 2)

    axs[row,col].bar(x, confidence_matrix[i])

    axs[row,col].set_title(questions[i])

    axs[row,col].set_ylim([4, 9])

    axs[row,col].set_xlabel('Rank')

    axs[row,col].set_ylabel('Confidence')

    

# Plot mean confidence score of each rank 

confidence_mean = np.mean(confidence_matrix, axis = 0) 

axs[5,0].bar(x, confidence_mean, color='orange')

axs[5,0].set_title("Average per rank") 

axs[5,0].set_ylim([4, 9])

axs[5,0].set_xlabel('Rank')

axs[5,0].set_ylabel('Confidence')



# Compare the confidence score of the 1st relevant article of each question

qs = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9", "Q10"]

axs[5,1].bar(qs, confidence_matrix[:,0], color='green')

axs[5,1].set_title("1st relevant articles with different questions") 

axs[5,1].set_ylim([4, 9])

axs[5,1].set_xlabel('Question #')

axs[5,1].set_ylabel('Confidence')



# Plot

fig.tight_layout(pad=2.0)