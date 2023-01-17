# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from __future__ import absolute_import, division, print_function

# __future__ is a pseudo-module which programmers can use to enable new language features which are not compatible with the current interpreter. 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import re

import copy

import json

import math



import six

import torch

import torch.nn as nn

from torch.nn import CrossEntropyLoss



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def gelu(x):

    """

    GELU: new activation function combining properties from dropout, zoneout, and ReLUs.

    source: https://arxiv.org/pdf/1606.08415.pdf

    

    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):

        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) 

        same as function given in previous paper

    """

    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
class BertConfig(object):

    """

    configuration class to store bert config

    """

    def __init__(self,

                vocab_size,

                hidden_size = 768,

                num_hidden_layers = 12,

                num_attention_heads = 12,

                intermediate_size = 3072,

                hidden_act = 'gelu',

                hidden_dropout_prob = 0.1,

                attention_probs_dropout_prob = 0.1,

                max_position_embeddings = 512,

                type_vocab_size = 16,

                initializer_range = 0.02):

        # default the uncased base model

        # intermediate size?

        """

        Args:

            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.

            hidden_size: Size of the encoder layers and the pooler layer.

            num_hidden_layers: Number of hidden layers in the Transformer encoder.

            num_attention_heads: Number of attention heads for each attention layer in

                the Transformer encoder.

            intermediate_size: The size of the "intermediate" (i.e., feed-forward)

                layer in the Transformer encoder.

            hidden_act: The non-linear activation function (function or string) in the

                encoder and pooler.

            hidden_dropout_prob: The dropout probabilitiy for all fully connected

                layers in the embeddings, encoder, and pooler.

            attention_probs_dropout_prob: The dropout ratio for the attention

                probabilities.

            max_position_embeddings: The maximum sequence length that this model might

                ever be used with. Typically set this to something large just in case

                (e.g., 512 or 1024 or 2048).

            type_vocab_size: The vocabulary size of the `token_type_ids` passed into

                `BertModel`.

            initializer_range: The sttdev of the truncated_normal_initializer for

                initializing all weight matrices.

        """

        self.vocab_size = vocab_size

        self.hidden_size = hidden_size

        self.num_hidden_layers = num_hidden_layers

        self.num_attention_heads = num_attention_heads

        self.hidden_act = hidden_act

        self.intermediate_size = intermediate_size

        self.hidden_dropout_prob = hidden_dropout_prob

        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        self.max_position_embeddings = max_position_embeddings

        self.type_vocab_size = type_vocab_size

        self.initializer_range = initializer_range

        

    @classmethod

    def from_dict(cls, json_object):

        """construct a BertConfig obj from python dictionary; assign parameters"""

        config = BertConfig(vocab_size = None)

        for (key, value) in six.iteritems(json_object):

            # This replaces dictionary.iteritems() on Python 2 and dictionary.items() on Python 3.

            config.__dict__[key] = value

#             A special attribute of every module is __dict__. This is the dictionary containing the module’s symbol table.

#             object.__dict__: A dictionary or other mapping object used to store an object’s (writable) attributes.

        return config



    @classmethod

    def from_json_file(cls, json_file):

        """construct a BertConfig obj from json file; assign parameters"""

        with open(json_file, 'r') as reader:

            text = reader.read()

        return cls.from_dict(json.loads(text))

    

    def to_dict(self):

        """serialize this instantce to a python dictionary"""

        output = copy.deepcopy(self.__dict__)

        return output

        

    def to_json_string(self):

        """serialize this instance to a json string; print parameters assigned"""

        # pretty printing

        return json.dumps(self.to_dict(), indent = 2, sort_keys = True) + '\n'

        
class BERTLayerNorm(nn.Module):

    def __init__(self, config, variance_epsilon = 1e-12):

        """

        construct a layerNorm module in the tf style (epsilon inside the square root) ???

        """

        super(BERTLayerNorm, self).__init__()

        # this is python2 way of writing super

        # python 3 way: super()

#         Base class for all neural network modules.

#         Your models should also subclass this class.



        self.gamma = nn.Parameter(torch.ones(config.hidden_size))

        self.beta = nn.Parameter(torch.zeros(config.hidden_size))

        # add to parameter list, can be accessed by .parameters()

        # nn.Parameters: a kind of tensor(multi dimnsional matrices) to be considered a module parameter

        self.variance_epsilon = variance_epsilon

        

    def forward(self, x):

        """

        layer:

        normalizing

        gamma * x + beta

        """

        u = x.mean(-1, keepdim = True)

        # if 2-dim matrix, column mean

        s = (x - u).pow(2).mean(-1, keepdim=True)

        # keep same dim as input except in the dimension -1

        x = (x - u) / torch.sqrt(s + self.variance_epsilon)

        return self.gamma * x + self.beta

        

class BERTEmbeddings(nn.Module):

    def __init__(self, config):

        super(BERTEmbeddings, self).__init__()

        """

        construct the embedding module from word, position and token_type embeddings

        """

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # contextural word embedding(from bert last layer) lookup table

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # position embedding lookup table

        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # token type embedding lookup table

        

        self.LayerNorm = BERTLayerNorm(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        

    def forward(self, input_ids, token_type_ids = None):

        """

        layer:

        get embedding

        norm layer

        drop out

        """

        # set parameters to same shape

        

        seq_length = input_ids.size(1)

        position_ids = torch.arange(seq_length, dtype = torch.long, device = input_ids.device)

        # returns a 1-d tensor of size seq_length, with values from the interval [0, seq_length)

        position_ids = position_ids.unsqeeze(0).expand_as(input_ids)

        # unsqueeze to a row, then expand to the same size as input_ids

        if token_type_ids is None:

            token_type_ids = torch.zeros_like(input_ids)

        

        # get the embeddings out of lookup tables

        

        words_embeddings = self.word_embeddings(input_ids)

        position_embeddings = self.positon_embeddings(position_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        # add all embedding together

        embeddings = self.LayerNorm(embeddings)

        # normalization

        embeddings = self.dropout(embeddings)

        # dropout for embedding layer

        return embeddings
class BERTSelfAttention(nn.Module):

    def __init__(self, config):

        super(BERTSelfAttention, self).__init__()

        """

        self attention

        """

        if config.hidden_size % config.num_attention_heads != 0:

            raise ValueError(

                "The hidden size (%d) is not a multiple of the number of attention "

                "heads (%d)" % (config.hidden_size, config.num_attention_heads)

            )

            # check para from config

        self.num_attention_heads = config.num_attention_heads

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)

        # single attention head size

        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # all attention head size

        

        self.query = nn.Linear(config.hidden_size, self.all_head_size)

        self.key = nn.Linear(config.hidden_size, self.all_head_size)

        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # apply linear transformation, input shape, output shape

        

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # dropout after attention layer

        

    def transpose_for_scores(self, x):

        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)

        x = x.view(*new_x_shape)

        # return a new tensor with the same data as the x but of shape of new_x_shape

        # since num_attention_heads * attention_head_size = hidden_size, the total num of values are the same

        return x.permute(0,2,1,3) 

        # the desired ordering of the dimensions

        

    def forward(self, hidden_states, attention_mask):

        ## linear

        

        mixed_query_layer = self.query(hidden_states)

        mixed_key_layer = self.key(hidden_states)

        mixed_value_layer = self.value(hidden_states)

        # get the linear output

        

        ## scaled dot-product attention

        

        query_layer = self.transpose_for_scores(mixed_query_layer)

        key_layer = self.transpose_for_scores(mixed_key_layer)

        value_layer = self.transpose_for_scores(mixed_value_layer)

        # change shape

        

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,-2))

        # take the dot product between query and key to get the raw attention score, the similarity between the two to obtain weights

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        # the attention mask is precomputed for all layers in BertModel forward() function

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # normalizing the attention scores to probabilities, normalize the weights

        attention_probs = self.dropout(attention_probs)

        # ??? this is actally dropping out entire tokens to attend to, which might seem a bit unusual, but is taken from the original transformer paper

        context_layer = torch.matmul(attention_prons, value_layer)

        # weight these weights in conjunction with the corresponding values to get final attention

        

        ## concat

        

        context_layer = context_layer.permute(0,2,1,3).contiguous()

#         Note that the word "contiguous" is bit misleading because its not that the content of tensor is spread out around disconnected blocks of memory. 

#         Here bytes are still allocated in one block of memory but the order of the elements is different!

#         When you call contiguous(), it actually makes a copy of tensor so the order of elements would be same as if tensor of same shape created from scratch.

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

        context_layer = context_layer.view(*new_context_layer_shape)

        # reshape into 2 dim

        return context_layer

        
class BERTSelfOutput(nn.Module):

    def __init__(self, config):

        super(BERTSelfOutput, self).__init__()

        """

        add & norm

        """

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.LayerNorm = BERTLayerNorm(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)



    def forward(self, hidden_states, input_tensor):

        hidden_states = self.dense(hidden_states)

        hidden_states = self.dropout(hidden_states)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
class BERTAttention(nn.Module):

    def __init__(self, config):

        super(BERTAttention, self).__init__()

        """

        self-attention + add&norm

        """

        self.self = BERTSelfAttention(config)

        self.output = BERTSelfOutput(config)



    def forward(self, input_tensor, attention_mask):

        self_output = self.self(input_tensor, attention_mask)

        attention_output = self.output(self_output, input_tensor)

        return attention_output
class BERTIntermediate(nn.Module):

    def __init__(self, config):

        super(BERTIntermediate, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

        self.intermediate_act_fn = gelu



    def forward(self, hidden_states):

        hidden_states = self.dense(hidden_states)

        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states
class BERTOutput(nn.Module):

    def __init__(self, config):

        super(BERTOutput, self).__init__()

        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

        self.LayerNorm = BERTLayerNorm(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)



    def forward(self, hidden_states, input_tensor):

        hidden_states = self.dense(hidden_states)

        hidden_states = self.dropout(hidden_states)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
class BERTLayer(nn.Module):

    def __init__(self, config):

        super(BERTLayer, self).__init__()

        """

        self-attention + add&norm

        + intermiediate + 

        feed forward + add&norm

        """

        self.attention = BERTAttention(config)

        self.intermediate = BERTIntermediate(config)

        self.output = BERTOutput(config)



    def forward(self, hidden_states, attention_mask):

        attention_output = self.attention(hidden_states, attention_mask)

        intermediate_output = self.intermediate(attention_output)

        layer_output = self.output(intermediate_output, attention_output)

        return layer_output
class BERTEncoder(nn.Module):

    def __init__(self, config):

        super(BERTEncoder, self).__init__()

        layer = BERTLayer(config)

        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])  

        # copy same structure of hidden layer



    def forward(self, hidden_states, attention_mask):

        all_encoder_layers = []

        for layer_module in self.layer:

            hidden_states = layer_module(hidden_states, attention_mask)

            all_encoder_layers.append(hidden_states)

            # stack all hidden layer outputs

        return all_encoder_layers

class BERTPooler(nn.Module):

    def __init__(self, config):

        super(BERTPooler, self).__init__()

        """???"""

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.activation = nn.Tanh()



    def forward(self, hidden_states):

        # We "pool" the model by simply taking the hidden state corresponding

        # to the first token.

        first_token_tensor = hidden_states[:, 0]

        #return first_token_tensor

        pooled_output = self.dense(first_token_tensor)

        pooled_output = self.activation(pooled_output)

        return pooled_output
class BertModel(nn.Module):

    """BERT model ("Bidirectional Embedding Representations from a Transformer").



    Example usage:

    ```python

    # Already been converted into WordPiece token ids

    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])

    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])

    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])



    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,

        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)



    model = modeling.BertModel(config=config)

    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)

    ```

    """

    def __init__(self, config: BertConfig):

        """Constructor for BertModel.



        Args:

            config: `BertConfig` instance.

        """

        super(BertModel, self).__init__()

        self.embeddings = BERTEmbeddings(config)

        self.encoder = BERTEncoder(config)

        self.pooler = BERTPooler(config)



    def forward(self, input_ids, token_type_ids=None, attention_mask=None):

        if attention_mask is None:

            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:

            token_type_ids = torch.zeros_like(input_ids)



        # We create a 3D attention mask from a 2D tensor mask.

        # Sizes are [batch_size, 1, 1, from_seq_length]

        # So we can broadcast to [batch_size, num_heads, to_seq_length, from_seq_length]

        # this attention mask is simpler than the triangular masking of causal attention

        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)



        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for

        # masked positions, this operation will create a tensor which is 0.0 for

        # positions we want to attend and -10000.0 for masked positions.

        # Since we are adding it to the raw scores before the softmax, this is

        # effectively the same as removing these entirely.

        extended_attention_mask = extended_attention_mask.float()

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0



        embedding_output = self.embeddings(input_ids, token_type_ids)

        all_encoder_layers = self.encoder(embedding_output, extended_attention_mask)

        sequence_output = all_encoder_layers[-1]

        # hidden states from last layer

        pooled_output = self.pooler(sequence_output)

        return all_encoder_layers, pooled_output
class BertForSequenceClassification(nn.Module):

    """BERT model for classification.

    This module is composed of the BERT model with a linear layer on top of

    the pooled output.



    Example usage:

    ```python

    # Already been converted into WordPiece token ids

    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])

    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])

    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])



    config = BertConfig(vocab_size=32000, hidden_size=512,

        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)



    num_labels = 2



    model = BertForSequenceClassification(config, num_labels)

    logits = model(input_ids, token_type_ids, input_mask)

    ```

    """

    def __init__(self, config, num_labels):

        super(BertForSequenceClassification, self).__init__()

        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, num_labels)



        def init_weights(module):

            """initialize parameters for layers"""

            if isinstance(module, (nn.Linear, nn.Embedding)):

                # Slightly different from the TF version which uses truncated_normal for initialization

                # cf https://github.com/pytorch/pytorch/pull/5617

                # initilize weight for linear and embedding

                module.weight.data.normal_(mean=0.0, std=config.initializer_range)

            elif isinstance(module, BERTLayerNorm):

                # initilize norm layer

                module.beta.data.normal_(mean=0.0, std=config.initializer_range)

                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)

            if isinstance(module, nn.Linear):

                # initialize bias for linear

                module.bias.data.zero_()

                

        self.apply(init_weights)



    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):

        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)



        if labels is not None:

            loss_fct = CrossEntropyLoss()

            loss = loss_fct(logits, labels)

            return loss, logits

        else:

            return logits
class BertForQuestionAnswering(nn.Module):

    """BERT model for Question Answering (span extraction).

    This module is composed of the BERT model with a linear layer on top of

    the sequence output that computes start_logits and end_logits



    Example usage:

    ```python

    # Already been converted into WordPiece token ids

    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])

    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])

    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])



    config = BertConfig(vocab_size=32000, hidden_size=512,

        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)



    model = BertForQuestionAnswering(config)

    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)

    ```

    """

    def __init__(self, config):

        super(BertForQuestionAnswering, self).__init__()

        self.bert = BertModel(config)

        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version

        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.qa_outputs = nn.Linear(config.hidden_size, 2)



        def init_weights(module):

            if isinstance(module, (nn.Linear, nn.Embedding)):

                # Slightly different from the TF version which uses truncated_normal for initialization

                # cf https://github.com/pytorch/pytorch/pull/5617

                module.weight.data.normal_(mean=0.0, std=config.initializer_range)

            elif isinstance(module, BERTLayerNorm):

                module.beta.data.normal_(mean=0.0, std=config.initializer_range)

                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)

            if isinstance(module, nn.Linear):

                module.bias.data.zero_()

        self.apply(init_weights)



    def forward(self, input_ids, token_type_ids, attention_mask, start_positions=None, end_positions=None):

        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)

        sequence_output = all_encoder_layers[-1]

        logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)

        end_logits = end_logits.squeeze(-1)

        # remove last dimension if it's 1



        if start_positions is not None and end_positions is not None:

            # If we are on multi-GPU, split add a dimension - if not this is a no-op

            start_positions = start_positions.squeeze(-1)

            end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms

            ignored_index = start_logits.size(1)

            start_positions.clamp_(0, ignored_index)

            end_positions.clamp_(0, ignored_index)

            # ???



            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            start_loss = loss_fct(start_logits, start_positions)

            end_loss = loss_fct(end_logits, end_positions)

            total_loss = (start_loss + end_loss) / 2

            return total_loss

        else:

            return start_logits, end_logits

conf = {

    "tf_checkpoint_cpkt": '/kaggle/input/bert-pretrain/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/bert_model.ckpt',

    "tf_checkpoint_meta": '/kaggle/input/bert-pretrain/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/bert_model.ckpt.meta',

    

    "bert_config_file": '/kaggle/input/bert-pretrain/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/bert_config.json',

    "pytorch_dump_path": '/kaggle/output'

}
# initialize pytorch model

config = BertConfig.from_json_file(conf['bert_config_file'])

model = BertModel(config)



# load weights from tf model

# when saving model in tf, there'd be files: meta/index/data. 

# meta describes the graph sturcture. tf.train.import_meta_graph(conf['tf_checkpoint_meta'])

# index has name-metadata of tensors

# data has valus of all variables

# even though there is no file named bert_model.ckpt, you still refer the the saved checkpoint by that name when restoring it

path = conf['tf_checkpoint_cpkt']

init_vars = tf.train.list_variables(path)

names = []

arrays = []

for name, shape in init_vars:

    print("Loading {} with shape {}".format(name, shape))

    array = tf.train.load_variable(path, name)

    print("Numpy array shape {}".format(array.shape))

    names.append(name)

    arrays.append(array)
for name, array in zip(names, arrays):

    name = name[5:] # skip 'bert/'

    print("Loading {}".format(name))

    name = name.split('/')

    if any(n in ['adam_v', 'adam_m', 'l_step'] for n in name):

        print("Skipping {}".format("/".join(name)))

        continue

    if name[0] in ['redictions', 'eq_relationship']:

        print("skipping")

        continue

    pointer = model

    # going down hierachicaly to get to the parameter

    for m_name in name:

        if re.fullmatch(r'[A-Za-z]+_\d+', m_name):

            l = re.split(r'_(\d+)', m_name)

        else:

            l = [m_name]

        if l[0] == 'kernel':

            pointer = getattr(pointer, 'weight')

        else:

             pointer = getattr(pointer, l[0])

        if len(l) >= 2:

            num = int(l[1])

            pointer = pointer[num]

    if m_name[-11:] == '_embeddings':

            pointer = getattr(pointer, 'weight')

    elif m_name == 'kernel':

        array = np.transpose(array)

    try:

        assert pointer.shape == array.shape

    except AssertionError as e:

        e.args += (pointer.shape, array.shape)

        raise

    # check and dump parameter value

    pointer.data = torch.from_numpy(array)



    

# save pytorch model

torch.save(model.state_dict(),'pytorch_model.bin')