# Remove any existing PyTorch so Kaggle will let us install a new one
! pip uninstall torch -y
! pip uninstall torch -y
## SimpleTransformers / PyTorch
! pip install simpletransformers
! pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
import torch
torch.__version__
import pandas as pd

news_train = pd.read_csv("/kaggle/input/tamil-nlp/tamil_news_train.csv", header=0, names=['article','headline','category','category_ta'])
news_train = news_train[['headline', 'category']]
news_train.category = pd.Categorical(news_train.category)
news_train['category'] = news_train['category'].cat.codes
news_train = news_train.dropna()
news_train.head()
news_test = pd.read_csv("/kaggle/input/tamil-nlp/tamil_news_test.csv", header=0, names=['article','headline','category','category_ta'])
news_test = news_test[['headline', 'category']]
news_test.category = pd.Categorical(news_test.category)
news_test['category'] = news_test['category'].cat.codes
news_test = news_test.dropna()
news_test.head()
len(news_train.category.unique())
from simpletransformers.classification import ClassificationModel
# set use_cuda=False on CPU-only platforms
model = ClassificationModel('bert', 'monsoon-nlp/tamillion', num_labels=6, use_cuda=True, args={
    'reprocess_input_data': True,
    'use_cached_eval_features': False,
    'overwrite_output_dir': True,
    'num_train_epochs': 3
})
model.train_model(news_train)
result, model_outputs, wrong_predictions = model.eval_model(news_test)
bads = {}
for pred in wrong_predictions:
    if pred.label in bads:
        bads[pred.label] += 1
    else:
        bads[pred.label] = 1
print("wrong predictions:")
print(str(len(wrong_predictions)) + ' wrong out of ' + str(len(news_test)))
bads
(3631-1028)/3631*100
## KTrain - TBD