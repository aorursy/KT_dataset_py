# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
! git clone https://github.com/NVIDIA/apex

! cd apex

! pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" /kaggle/working/apex/
!pip install simpletransformers
from simpletransformers.classification import ClassificationModel
# Lets import the csv file in pandas dataframe first

train_df = pd.read_csv('/kaggle/input/ttc4900/7all.csv', encoding='utf-8', header=None, names=['cat', 'text'])
# Check the df

train_df.head()
# unique categories

print(train_df.cat.unique())

print("Total categories",len(train_df.cat.unique()))

# convert string labels to integers

train_df['labels'] = pd.factorize(train_df.cat)[0]



train_df.head()
# Let's create a train and test set

from sklearn.model_selection import train_test_split



train, test = train_test_split(train_df, test_size=0.2, random_state=42)
train.shape, test.shape
# Lets define the model with the parameters (important here is the number of labels and nr of epochs)



model = ClassificationModel('bert', 'bert-base-multilingual-uncased', num_labels=7, 

                            args={'reprocess_input_data': True, 'overwrite_output_dir': True, 'num_train_epochs': 3})
# Now lets fine tune bert with the train set

model.train_model(train)
# Let's evaluate this finetuned model with the test set

result, model_outputs, wrong_predictions = model.eval_model(test)
predictions = model_outputs.argmax(axis=1)
predictions[0:10]
actuals = test.labels.values

actuals[0:10]
# Now lets see the accuracy one the test set

from sklearn.metrics import accuracy_score

accuracy_score(actuals, predictions)
sample_text = test.iloc[10]['text']

print(sample_text)
# Lets predict the text of sample_text:

model.predict([sample_text])
# Lets see what the truth was

test.iloc[10]['labels']

# And this was category: 

test.iloc[10]['cat']