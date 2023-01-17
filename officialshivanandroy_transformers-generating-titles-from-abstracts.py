! pip uninstall torch torchvision -y
! pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
!pip install -U transformers
!pip install -U simpletransformers  
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os, psutil  

def cpu_stats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    return 'memory GB:' + str(np.round(memory_use, 2))
cpu_stats()
import json

data_file = '../input/arxiv/arxiv-metadata-oai-snapshot.json'

def get_metadata():
    with open(data_file, 'r') as f:
        for line in f:
            yield line
metadata = get_metadata()
for paper in metadata:
    paper_dict = json.loads(paper)
    print('Title: {}\n\nAbstract: {}\nRef: {}'.format(paper_dict.get('title'), paper_dict.get('abstract'), paper_dict.get('journal-ref')))
#     print(paper)
    break
titles = []
abstracts = []
years = []
metadata = get_metadata()
for paper in metadata:
    paper_dict = json.loads(paper)
    ref = paper_dict.get('journal-ref')
    try:
        year = int(ref[-4:]) 
        if 2016 < year < 2021:
            years.append(year)
            titles.append(paper_dict.get('title'))
            abstracts.append(paper_dict.get('abstract'))
    except:
        pass 

len(titles), len(abstracts), len(years)
papers = pd.DataFrame({
    'title': titles,
    'abstract': abstracts,
    'year': years
})
papers.head()
del titles, abstracts, years
cpu_stats()
import logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

papers = papers[['title','abstract']]
papers.columns = ['target_text', 'input_text']
papers = papers.dropna()
eval_df = papers.sample(frac=0.2, random_state=101)
train_df = papers.drop(eval_df.index)
train_df.shape, eval_df.shape
import logging

import pandas as pd
from simpletransformers.t5 import T5Model

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_df['prefix'] = "summarize"
eval_df['prefix'] = "summarize"


model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 512,
    "train_batch_size": 16,
    "num_train_epochs": 4,
}

# Create T5 Model
model = T5Model("t5-small", args=model_args, use_cuda=True)

# Train T5 Model on new task
model.train_model(train_df)

# Evaluate T5 Model on new task
results = model.eval_model(eval_df)

# Predict with trained T5 model
#print(model.predict(["convert: four"]))
results
random_num = 350
actual_title = eval_df.iloc[random_num]['target_text']
actual_abstract = ["summarize: "+eval_df.iloc[random_num]['input_text']]
predicted_title = model.predict(actual_abstract)

print(f'Actual Title: {actual_title}')
print(f'Predicted Title: {predicted_title}')
print(f'Actual Abstract: {actual_abstract}')

random_num = 478
actual_title = eval_df.iloc[random_num]['target_text']
actual_abstract = ["summarize: "+eval_df.iloc[random_num]['input_text']]
predicted_title = model.predict(actual_abstract)

print(f'Actual Title: {actual_title}')
print(f'Predicted Title: {predicted_title}')
print(f'Actual Abstract: {actual_abstract}')
random_num = 999
actual_title = eval_df.iloc[random_num]['target_text']
actual_abstract = ["summarize: "+eval_df.iloc[random_num]['input_text']]
predicted_title = model.predict(actual_abstract)

print(f'Actual Title: {actual_title}')
print(f'Predicted Title: {predicted_title}')
print(f'Actual Abstract: {actual_abstract}')