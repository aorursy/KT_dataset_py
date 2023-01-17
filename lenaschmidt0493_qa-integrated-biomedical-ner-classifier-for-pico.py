#Clone and install sources from GIT
!pip install git+https://github.com/L-ENA/transformers
!git clone https://github.com/L-ENA/CORD19scripts    

###general imports
import os
import re
import pandas as pd
import json
import numpy as np
import collections
from collections import Counter
from collections import defaultdict
from glob import glob
from tqdm import tqdm
import random

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

from IPython.display import display, HTML
pd.set_option('display.max_rows', 5)


##set up kaggle working dirs
os.makedirs('/kaggle/working/predictions/')
os.makedirs('/kaggle/working/predictions/plots/')
os.makedirs('/kaggle/working/train_test_pred_data/')
#import my scripts and functions and process data for the interventions
from CORD19scripts import covid_input

#The methods for obtaining these predicions are described in this Notebook's Methods section on a high level, and they will be decribed in-depth for every separate task submission
#do something with the predictions: first we need to post-processs them a tiny bit, then link the to the original dataset and sort them by frequency
print("Processing Condition predictions......")
covid_input.connectInput(["/kaggle/working/CORD19scripts/predictions/outputs"], mode="Condition")   #modes "Condition" or else "Population"for Patints or else "Intervention"  #to make unified csv file

df = pd.read_csv("/kaggle/working/predictions/predictionsLENA_C.csv")#show the data
display(df[:5])
#The we run a simple unsupervised clustering based on substrings in the mined data
print("Deduplicating and clustering Intervention predictions......")
covid_input.deduplicate_predictions("predictions/predictionsLENA_C.csv", mode="C")

df = pd.read_csv("/kaggle/working/predictions/C_deduped.csv")
display(df[:5])
from CORD19scripts import interactive_plots
    
#Params:
#
#"n_entries = 5" means that we will display 5*5=25 top mining results in the interactive plot
#"mode="C" means that we will look at C(ondition)-type entities

interactive_plots.make_plot_kaggle(n_entries = 5, mode="C") 
#do something with the predictions: first we need to post-processs them a tiny bit, then link the to the original dataset and sort them by frequency
print("Processing Intervention predictions......")
covid_input.connectInput(["/kaggle/working/CORD19scripts/predictions/outputs"], mode="Intervention")   #modes "Condition" or else "Population"for Patints or else "Intervention"  #to make unified csv file

#The we run a simple unsupervised clustering based on substrings in the mined data
print("Deduplicating and clustering Intervention predictions......")
covid_input.deduplicate_predictions("predictions/predictionsLENA_I.csv", mode="I")
#Params:
#
#"n_entries = 5" means that we will display 5*5=25 top mining results in the interactive plot - feel free to change this and to zoom in and out of the plots
#"mode="I" means that we will look at I(ntervention)-type entities

interactive_plots.make_plot_kaggle(n_entries = 5, mode="I") 
from CORD19scripts import make_train_test


ebmnlp_path="/kaggle/input/ebmnlp/ebm_nlp_2_00/ebm_nlp_2_00"#path to annotated data
squad_path="/kaggle/input/squad-20"#path to squad data

#Create the training and evaluation data that will be used to fine-tune the Transformer model!
make_train_test.make_data(ebmnlp_path,squad_path, entity="I", makeTest=True, add_dev=0)#make the testing data
make_train_test.make_data(ebmnlp_path,squad_path, entity="I", makeTest=False, undersample_frac = 0.3, add_dev=0,add_train=170)#make and save the file in this notebooks output directory, can be downloaded on the right sidebar under "/kaggle/working/train_test_pred_data"




from CORD19scripts import covid_input

#just slightly adjusting the cord dataframe to fit my previous scripts. 
#Also, we are selecting only publications after 2020 when using the "min_date=2020" parameter
covid_input.rewrite_cord_data("/kaggle/input/CORD-19-research-challenge/metadata.csv", max_rows=150000, min_date=2020)

#saves 'sent_map.csv'to the kaggle output predictions folder, and the data to predict to "train_test_pred_data" folder
#if we give types="I" as parameter, like below,  then it will make a file called "dev-v2.0_cord_pred_I.json", indicating that this is:
#a "dev" file for the v2 SQuAD task
#that it is based on the "cord" data
#and that it is for entity type "I"; 

#NB when feeding this file to the network its easiest to just rename it like the original squad dev file: "dev-v2.0.json"
covid_input.makeInput(["/kaggle/working/covid_data.csv"], types="I")



