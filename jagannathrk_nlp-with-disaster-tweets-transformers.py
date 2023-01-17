!pip install simpletransformers
import os, re, string

import random



import numpy as np

import pandas as pd

import sklearn



import torch



from simpletransformers.classification import ClassificationModel

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
seed = 1337



random.seed(seed)

np.random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Zero pre-processing of training data

# I have tried doing some preprcessing/cleaning but the result 

# does not seem significant

train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

train_data = train_data[['text', 'target']]
# Using 'bert-large-uncased' here. For a list of other models, please refer to 

# https://github.com/ThilinaRajapakse/simpletransformers/#current-pretrained-models 

bert_uncased = ClassificationModel('bert', 'bert-large-uncased') 



# Print out all the default arguments for reference

bert_uncased.args
# This is where we can tweak based on the default arguments above

custom_args = {'fp16': False, # not using mixed precision 

               'train_batch_size': 4, # default is 8

               'gradient_accumulation_steps': 2,

               'do_lower_case': True,

               'learning_rate': 1e-05, # using lower learning rate

               'overwrite_output_dir': True, # important for CV

               'num_train_epochs': 2} # default is 1
n=5

kf = KFold(n_splits=n, random_state=seed, shuffle=True)

results = []



for train_index, val_index in kf.split(train_data):

    train_df = train_data.iloc[train_index]

    val_df = train_data.iloc[val_index]

    

    model = ClassificationModel('bert', 'bert-base-uncased', args=custom_args) 

    model.train_model(train_df)

    result, model_outputs, wrong_predictions = model.eval_model(val_df, acc=sklearn.metrics.accuracy_score)

    print(result['acc'])

    results.append(result['acc'])
for i, result in enumerate(results, 1):

    print(f"Fold-{i}: {result}")

    

print(f"{n}-fold CV accuracy result: Mean: {np.mean(results)} Standard deviation:{np.std(results)}")
model = ClassificationModel('bert', 'bert-base-uncased', args=custom_args) 

model.train_model(train_data)
test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

predictions, raw_outputs = model.predict(test_data['text'])



sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = predictions

sample_submission.to_csv("submission.csv", index=False)