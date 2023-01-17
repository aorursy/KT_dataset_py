# Python 

import os

import warnings

import logging

from typing import Mapping, List, Union, Optional, Tuple

from pprint import pprint



# Numpy, Pandas, Sklearn

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split



# PyTorch 

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader



# Transformers 

from transformers import AutoConfig, AutoModel, AutoTokenizer



# Catalyst

import catalyst

from catalyst.dl import SupervisedRunner

# this will appear in Catalyst 20.08

# from catalyst.dl.callbacks.metrics.accuracy import MultiLabelAccuracyCallback

from catalyst.dl.callbacks import OptimizerCallback, CheckpointCallback, InferCallback

from catalyst.dl.utils import plot_metrics

from catalyst.utils import set_global_seed, prepare_cudnn
catalyst.__version__
MODEL_NAME = 'distilbert-base-uncased' # pretrained model from Transformers

LOG_DIR = "./logdir"                   # for training logs and tensorboard visualizations

NUM_EPOCHS = 3                         # smth around 2-6 epochs is typically fine when finetuning transformers

BATCH_SIZE = 96                        # depends on your available GPU memory (in combination with max seq length)

MAX_SEQ_LENGTH = 256                   # depends on your available GPU memory (in combination with batch size)

LEARN_RATE = 3e-5                      # learning rate is typically ~1e-5 for transformers

ACCUM_STEPS = 4                        # one optimization step for that many backward passes

SEED = 17                              # random seed for reproducibility
# to reproduce, download the data and customize this path

PATH_TO_DATA = '../input/jigsaw-toxic-comment-classification-challenge/'

TEXT_FIELD = 'comment_text'

TARGET_FIELDS = ['toxic','severe_toxic','obscene','threat','insult', 'identity_hate']

NUM_CLASSES = len(TARGET_FIELDS)

PRED_THRES = 0.4   
train_df = pd.read_csv(PATH_TO_DATA + 'train.csv.zip', index_col='id')

test_df = pd.read_csv(PATH_TO_DATA + 'test.csv.zip', index_col='id')
train_df.info()
test_df.info()
train_df.head(2)
X_train, X_valid, y_train, y_valid = train_test_split(train_df[TEXT_FIELD],

                                                      train_df[TARGET_FIELDS], 

                                                      test_size=0.1, 

                                                      random_state=17)

X_test = test_df[TEXT_FIELD]
len(X_train), len(X_valid), len(X_test)
class TextClassificationDataset(Dataset):

    """

    Wrapper around Torch Dataset to perform text classification

    """

    def __init__(self,

                 texts: List[str],

                 labels: np.ndarray = None,

                 max_seq_length: int = 512,

                 model_name: str = 'distilbert-base-uncased'):

        """

        Args:

            texts (List[str]): a list with texts to classify or to train the

                classifier on

            labels List[str]: 

            max_seq_length (int): maximal sequence length in tokens,

                texts will be stripped to this length

            model_name (str): transformer model name, needed to perform

                appropriate tokenization



        """



        self.texts = texts

        self.labels = labels

        self.max_seq_length = max_seq_length



        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # suppresses tokenizer warnings

        logging.getLogger(

            "transformers.tokenization_utils").setLevel(logging.FATAL)



    def __len__(self):

        """

        Returns:

            int: length of the dataset

        """

        return len(self.texts)



    def __getitem__(self, index) -> Mapping[str, torch.Tensor]:

        """Gets element of the dataset



        Args:

            index (int): index of the element in the dataset

        Returns:

            Single element by index

        """



        # encoding the text

        x = self.texts[index]

        

        # a dictionary with `input_ids` and `attention_mask` as keys

        output_dict = self.tokenizer.encode_plus(

            x,

            add_special_tokens=True,

            pad_to_max_length=True,

            max_length=self.max_seq_length,

            return_tensors="pt",

            return_attention_mask=True

        )

        

        # for Catalyst, there needs to be a key called features

        output_dict['features'] = output_dict['input_ids'].squeeze(0)

        del output_dict['input_ids']

        

        # encoding target

        if self.labels is not None:

            output_dict["targets"] = torch.from_numpy(self.labels[index]).float()



        return output_dict
train_dataset = TextClassificationDataset(

    texts=X_train.values.tolist(),

    labels=y_train.values,

    max_seq_length=MAX_SEQ_LENGTH,

    model_name=MODEL_NAME

)



valid_dataset = TextClassificationDataset(

    texts=X_valid.values.tolist(),

    labels=y_valid.values,

    max_seq_length=MAX_SEQ_LENGTH,

    model_name=MODEL_NAME

)



test_dataset = TextClassificationDataset(

    texts=X_test.values.tolist(),

    max_seq_length=MAX_SEQ_LENGTH,

    model_name=MODEL_NAME

)
train_df.iloc[0]
pprint(train_dataset[0])
train_val_loaders = {

    "train": DataLoader(dataset=train_dataset,

                        batch_size=BATCH_SIZE, 

                        shuffle=True),

    "valid": DataLoader(dataset=valid_dataset,

                        batch_size=BATCH_SIZE, 

                        shuffle=False)    

}
class BertForSequenceClassification(nn.Module):

    """

    Simplified version of the same class by HuggingFace.

    See transformers/modeling_distilbert.py in the transformers repository.

    """



    def __init__(self, pretrained_model_name: str, num_classes: int = None, dropout: float = 0.3):

        """

        Args:

            pretrained_model_name (str): HuggingFace model name.

                See transformers/modeling_auto.py

            num_classes (int): the number of class labels

                in the classification task

        """

        super().__init__()



        config = AutoConfig.from_pretrained(

            pretrained_model_name, num_labels=num_classes)



        self.model = AutoModel.from_pretrained(pretrained_model_name,

                                                    config=config)

#         self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, num_classes)

        self.dropout = nn.Dropout(dropout)



    def forward(self, features, attention_mask=None, head_mask=None):

        """Compute class probabilities for the input sequence.



        Args:

            features (torch.Tensor): ids of each token,

                size ([bs, seq_length]

            attention_mask (torch.Tensor): binary tensor, used to select

                tokens which are used to compute attention scores

                in the self-attention heads, size [bs, seq_length]

            head_mask (torch.Tensor): 1.0 in head_mask indicates that

                we keep the head, size: [num_heads]

                or [num_hidden_layers x num_heads]

        Returns:

            PyTorch Tensor with predicted class probabilities

        """

        assert attention_mask is not None, "attention mask is none"

        

        bert_output = self.model(input_ids=features,

                                            attention_mask=attention_mask,

                                            head_mask=head_mask)

        # we only need the hidden state here and don't need

        # transformer output, so index 0

        seq_output = bert_output[0]  # (bs, seq_len, dim)

        # mean pooling, i.e. getting average representation for all tokens

        pooled_output = seq_output.mean(axis=1)  # (bs, dim)

        pooled_output = self.dropout(pooled_output)  # (bs, dim)

        logits = self.classifier(pooled_output)  # (bs, dim)



        return logits
model = BertForSequenceClassification(pretrained_model_name=MODEL_NAME,

                                      num_classes=NUM_CLASSES)
# d = next(iter(train_val_loaders['train']))

# p = model(d['features'], d['attention_mask'])

# criterion(p, d['targets'])
criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
from catalyst.core import MetricCallback

from catalyst.utils.torch import get_activation_fn



def preprocess_multi_label_metrics(

    outputs: torch.Tensor,

    targets: torch.Tensor,

    weights: Optional[torch.Tensor] = None,

) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    """

    General preprocessing and check for multi-label-based metrics.

    Args:

        outputs (torch.Tensor): NxK tensor that for each of the N examples

            indicates the probability of the example belonging to each of

            the K classes, according to the model.

        targets (torch.Tensor): binary NxK tensor that encodes which of the K

            classes are associated with the N-th input

            (eg: a row [0, 1, 0, 1] indicates that the example is

            associated with classes 2 and 4)

        weights (torch.Tensor): importance for each sample

    Returns:

        processed ``outputs`` and ``targets``

        with [batch_size; num_classes] shape

    """

    if not torch.is_tensor(outputs):

        outputs = torch.from_numpy(outputs)

    if not torch.is_tensor(targets):

        targets = torch.from_numpy(targets)

    if weights is not None:

        if not torch.is_tensor(weights):

            weights = torch.from_numpy(weights)

        weights = weights.squeeze()



    if outputs.dim() == 1:

        outputs = outputs.view(-1, 1)

    else:

        assert outputs.dim() == 2, (

            "wrong `outputs` size "

            "(should be 1D or 2D with one column per class)"

        )



    if targets.dim() == 1:

        targets = targets.view(-1, 1)

    else:

        assert targets.dim() == 2, (

            "wrong `targets` size "

            "(should be 1D or 2D with one column per class)"

        )



    if weights is not None:

        assert weights.dim() == 1, "Weights dimension should be 1"

        assert weights.numel() == targets.size(

            0

        ), "Weights dimension 1 should be the same as that of target"

        assert torch.min(weights) >= 0, "Weight should be non-negative only"



    assert torch.equal(

        targets ** 2, targets

    ), "targets should be binary (0 or 1)"



    return outputs, targets, weights



def multi_label_accuracy(

    outputs: torch.Tensor,

    targets: torch.Tensor,

    threshold: Union[float, torch.Tensor],

    activation: Optional[str] = None,

) -> torch.Tensor:

    """

    Computes multi-label accuracy for the specified activation and threshold.

    Args:

        outputs (torch.Tensor): NxK tensor that for each of the N examples

            indicates the probability of the example belonging to each of

            the K classes, according to the model.

        targets (torch.Tensor): binary NxK tensort that encodes which of the K

            classes are associated with the N-th input

            (eg: a row [0, 1, 0, 1] indicates that the example is

            associated with classes 2 and 4)

        threshold (float): threshold for for model output

        activation (str): activation to use for model output

    Returns:

        computed multi-label accuracy

    """

    outputs, targets, _ = preprocess_multi_label_metrics(

        outputs=outputs, targets=targets

    )

    activation_fn = get_activation_fn(activation)

    outputs = activation_fn(outputs)



    outputs = (outputs > threshold).long()

    output = (targets.long() == outputs.long()).sum().float() / np.prod(

        targets.shape

    )

    return output





class MultiLabelAccuracyCallback(MetricCallback):

    """Accuracy metric callback.

    Computes multi-class accuracy@topk for the specified values of `topk`.

    .. note::

        For multi-label accuracy please use

        `catalyst.dl.callbacks.metrics.MultiLabelAccuracyCallback`

    """



    def __init__(

        self,

        input_key: str = "targets",

        output_key: str = "logits",

        prefix: str = "multi_label_accuracy",

        threshold: float = None,

        activation: str = "Sigmoid",

    ):

        """

        Args:

            input_key (str): input key to use for accuracy calculation;

                specifies our `y_true`

            output_key (str): output key to use for accuracy calculation;

                specifies our `y_pred`

            prefix (str): key for the metric's name

            threshold (float): threshold for for model output

            activation (str): An torch.nn activation applied to the outputs.

                Must be one of ``"none"``, ``"Sigmoid"``, or ``"Softmax"``

        """

        super().__init__(

            prefix=prefix,

            metric_fn=multi_label_accuracy,

            input_key=input_key,

            output_key=output_key,

            threshold=threshold,

            activation=activation,

        )
os.environ['CUDA_VISIBLE_DEVICES'] = "0"    # can be changed in case of multiple GPUs onboard

set_global_seed(SEED)                       # reproducibility

prepare_cudnn(deterministic=True)           # reproducibility
%%time

# here we specify that we pass masks to the runner. So model's forward method will be called with

# these arguments passed to it. 

runner = SupervisedRunner(

    input_key=(

        "features",

        "attention_mask"

    )

)





# model training

runner.train(

    model=model,

    criterion=criterion,

    optimizer=optimizer,

    scheduler=scheduler,

    loaders=train_val_loaders,

    callbacks=[

        MultiLabelAccuracyCallback(threshold=PRED_THRES),

        OptimizerCallback(accumulation_steps=ACCUM_STEPS)

    ],

    logdir=LOG_DIR,

    num_epochs=NUM_EPOCHS,

    verbose=True

)
!du -hc $LOG_DIR
!ls $LOG_DIR/checkpoints
!nvidia-smi
torch.cuda.empty_cache()
!nvidia-smi
plot_metrics(

    logdir=LOG_DIR,

    step='epoch',

    metrics=['accuracy']

)
test_loaders = {

    "test": DataLoader(dataset=test_dataset,

                        batch_size=BATCH_SIZE, 

                        shuffle=False) 

}
%%time

runner.infer(

    model=model,

    loaders=test_loaders,

    callbacks=[

        CheckpointCallback(

            resume=f"{LOG_DIR}/checkpoints/best.pth"

        ),

        InferCallback(),

    ],   

    verbose=True

)
predicted_probs = runner.callbacks[0].predictions['logits']
predicted_probs.shape
sample_sub_df = pd.read_csv(PATH_TO_DATA + 'sample_submission.csv.zip',

                           index_col='id')
sample_sub_df.head(2)
sample_sub_df[TARGET_FIELDS] = predicted_probs
sample_sub_df.to_csv('submissions.csv')
!head -3 submissions.csv