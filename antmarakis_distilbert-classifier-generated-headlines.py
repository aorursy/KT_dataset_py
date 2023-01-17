!pip install transformers

!pip install simpletransformers
!mkdir data
import pandas as pd



train = pd.read_csv('/kaggle/input/generated-headlines/data_train.csv').sample(frac=1)

test = pd.read_csv('/kaggle/input/generated-headlines/data_test.csv').sample(frac=1)



train = train.rename(columns={'headline': 'text', 'fake': 'label'})

test = test.rename(columns={'headline': 'text', 'fake': 'label'})



train['text'] = train['text'].astype(str)

test['text'] = test['text'].astype(str)



train = train[['text', 'label']]

test = test[['text', 'label']]
test.dtypes
from simpletransformers.model import TransformerModel
from simpletransformers.model import TransformerModel
from simpletransformers.model import TransformerModel
model = TransformerModel('distilbert', 'distilbert-base-uncased-distilled-squad',

                         args={'fp16':False, 'train_batch_size':64, 'eval_batch_size':64, 'max_seq_length':20})
model.train_model(train)
preds, _ = model.predict(test['text'])
import numpy as np



def accuracy_percentile(preds, Y_validate):

    """Return the percentage of correct predictions for each class and in total"""

    real_correct, fake_correct, total_correct = 0, 0, 0

    _, (real_count, fake_count) = np.unique(Y_validate, return_counts=True)



    for i, r in enumerate(preds):

        if r == Y_validate[i]:

            total_correct += 1

            if r == 1:

                fake_correct += 1

            else:

                real_correct += 1



    print('Real Accuracy:', real_correct/real_count * 100, '%')

    print('Fake Accuracy:', fake_correct/fake_count * 100, '%')

    print('Total Accuracy:', total_correct/(real_count + fake_count) * 100, '%')
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc



accuracy_percentile(preds, list(test['label']))

fpr, tpr, _ = roc_curve(preds, test['label'])

print('AUC: {}'.format(auc(fpr, tpr)))
print('Accuracy: {}'.format(accuracy_score(preds, test['label'])))

print('Precision: {}'.format(precision_score(preds, test['label'])))

print('Recall: {}'.format(recall_score(preds, test['label'])))

print('F1: {}'.format(f1_score(preds, test['label'])))
print('Precision: {}'.format(precision_score(preds, test['label'], average='macro')))

print('Recall: {}'.format(recall_score(preds, test['label'], average='macro')))

print('F1: {}'.format(f1_score(preds, test['label'], average='macro')))