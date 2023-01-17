import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import transformers

from transformers import (

    AutoConfig,

    AutoTokenizer,

    AutoModel,

    TFAutoModel,

    AutoModelWithLMHead,

    TFAutoModelWithLMHead,

    AutoModelForSequenceClassification,

    TFAutoModelForSequenceClassification,

    AutoModelForQuestionAnswering,

    TFAutoModelForQuestionAnswering,

    AutoModelForTokenClassification,

    TFAutoModelForTokenClassification

)



# We will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



print(transformers.__version__)
MODEL_DIR = '/kaggle/input/roberta-transformers-pytorch/roberta-base/' # Adapt this line to the model directory of your choice

config =  AutoConfig.from_pretrained(MODEL_DIR)

print(config)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModel.from_pretrained(MODEL_DIR) # Comment this line if TF2.0 model

# model = TFAutoModel.from_pretrained(MODEL_DIR) # Uncomment this line if TF2.0 model

print(model)
model_for_sequence_classification = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR) # Comment this line if TF2.0 model

# model_for_sequence_classification = TFAutoModelForSequenceClassification.from_pretrained(MODEL_DIR) # Uncomment this line if TF2.0 model

print(model_for_sequence_classification)