!pip install editdistance==0.4

!pip install torch==0.4.0
import json 



import numpy as np

import torch



import os



from torch.utils import data

from random import choice, randrange

from itertools import zip_longest



import editdistance

import tqdm



%matplotlib inline

import matplotlib.pyplot as plt

plt.switch_backend('agg')

import matplotlib.ticker as ticker

import numpy as np
config=None

with open("../input/example_luong.json", "r") as f:

    config = json.load(f)

config
import mock

FLAGS = mock.Mock()

FLAGS.config = "example_luong.json"

FLAGS.epochs = 20

FLAGS.train_size = 28000

FLAGS.eval_size = 2600
import librosa

import numpy as np

import editdistance

import torch



EOS_TOKEN = '</s>'





def check_size(tensor, *args):

    size = [a for a in args]

    assert tensor.size() == torch.Size(size), tensor.size()



def to_mono(y):

    assert y.ndim == 2

    return np.mean(y, axis=1)





def downsample(y, orig_sr, targ_sr):

    if y.dtype != np.float:

        y = y.astype(np.float32)

    return librosa.resample(y, orig_sr=orig_sr, target_sr=targ_sr)





def standardize(x):

    new_x = (x - np.mean(x, 0)) / (np.std(x, 0) + 1e-3)

    return new_x





def edit_distance(guess, truth):

    guess = guess.split(EOS_TOKEN)[0]

    truth = truth[3:].split(EOS_TOKEN)[0]

    return editdistance.eval(guess, truth) / len(truth)





class AttrDict(dict):

  __getattr__ = dict.__getitem__

  __setattr__ = dict.__setitem__
class ToyDataset(data.Dataset):

    """

    https://talbaumel.github.io/blog/attention/

    """

    def __init__(self, min_length=5, max_length=20,type='train',size=3000):

        self.SOS = "<s>"  # all strings start  with the Beginning Of Stream token

        self.EOS = "</s>"  # all strings will end with the End Of Stream token

        self.characters = list("abcd")

        self.int2char = list(self.characters)

        self.char2int = {c: i+3 for i, c in enumerate(self.characters)}

        print(self.char2int)

        self.VOCAB_SIZE = len(self.characters)

        self.min_length = min_length

        self.max_length = max_length

        if type == 'train':

            self.set = [self._sample() for _ in range(size)]

        else:

            self.set = [self._sample() for _ in range(int(size/10))]



    def __len__(self):

        return len(self.set)



    def __getitem__(self, item):

        return self.set[item]



    def _sample(self):

        random_length = randrange(self.min_length, self.max_length)  # Pick a random length

        random_char_list = [choice(self.characters[:-1]) for _ in range(random_length)]  # Pick random chars

        random_string = ''.join(random_char_list)

        a = np.array([self.char2int.get(x) for x in random_string])

        b = np.array([self.char2int.get(x) for x in random_string[::-1]] + [2]) # Return the random string and its reverse

        x = np.zeros((random_length, self.VOCAB_SIZE))



        x[np.arange(random_length), a-3] = 1



        return x, b
dataset_train = ToyDataset(5, 15)

dataset_eval = ToyDataset(5, 15,"eval")

dataset_train.__len__() , dataset_eval.__len__()
char2int_dict = dataset_train.char2int

int2char_dict = {v: k for k, v in char2int_dict.items()}

chars = list(char2int_dict.keys())

chars,char2int_dict,int2char_dict
sample = dataset_train._sample()

print(sample)

input_sample = sample[0]

output_sample = sample[1]

input_chars = []

for s in input_sample:

    ix, = np.where(s == 1)

    input_chars.append(chars[ix[0]])

output_chars = [int2char_dict[s] for s in output_sample if s in int2char_dict.keys()]

    

input_chars,output_chars,input_chars==output_chars[::-1]
def batch(iterable, n=1):

    args = [iter(iterable)] * n

    return zip_longest(*args)





def pad_tensor(vec, pad, value=0, dim=0):

    """

    args:

        vec - tensor to pad

        pad - the size to pad to

        dim - dimension to pad

    return:

        a new tensor padded to 'pad' in dimension 'dim'

    """

    pad_size = pad - vec.shape[0]



    if len(vec.shape) == 2:

        zeros = torch.ones((pad_size, vec.shape[-1])) * value

    elif len(vec.shape) == 1:

        zeros = torch.ones((pad_size,)) * value

    else:

        raise NotImplementedError

    return torch.cat([torch.Tensor(vec), zeros], dim=dim)





def pad_collate(batch, values=(0, 0), dim=0):

    """

    args:

        batch - list of (tensor, label)

    return:

        xs - a tensor of all examples in 'batch' after padding

        ys - a LongTensor of all labels in batch

        ws - a tensor of sequence lengths

    """



    sequence_lengths = torch.Tensor([int(x[0].shape[dim]) for x in batch])

    sequence_lengths, xids = sequence_lengths.sort(descending=True)

    target_lengths = torch.Tensor([int(x[1].shape[dim]) for x in batch])

    target_lengths, yids = target_lengths.sort(descending=True)

    # find longest sequence

    src_max_len = max(map(lambda x: x[0].shape[dim], batch))

    tgt_max_len = max(map(lambda x: x[1].shape[dim], batch))

    # pad according to max_len

    batch = [(pad_tensor(x, pad=src_max_len, dim=dim), pad_tensor(y, pad=tgt_max_len, dim=dim)) for (x, y) in batch]



    # stack all

    xs = torch.stack([x[0] for x in batch], dim=0)

    ys = torch.stack([x[1] for x in batch]).int()

    xs = xs[xids]

    ys = ys[yids]

    return xs, ys, sequence_lengths.int(), target_lengths.int()



for item in batch(iter([1,2,3,4,5,6,7,8,9,10]),4):

    print(item)
tensor = torch.tensor([[1.,2.,3.],[4.,5.,6.]])

pad_tensor(tensor,pad=4,dim=1)
tensor = [(torch.tensor([[1.,2.,3.],[1.,2.,3.]]),torch.tensor([1.])),(torch.tensor([[4.,5.,6.]]),torch.tensor([0.,1.])),(torch.tensor([[7.,8.,9.]]),torch.tensor([0.]))]

pad_collate(tensor)
def mask_3d(inputs, seq_len, mask_value=0.):

    batches = inputs.size()[0]

    assert batches == len(seq_len)

    max_idx = max(seq_len)

    for n, idx in enumerate(seq_len):

        if idx < max_idx.item():

            if len(inputs.size()) == 3:

                inputs[n, idx.int():, :] = mask_value

            else:

                assert len(inputs.size()) == 2, "The size of inputs must be 2 or 3, received {}".format(inputs.size())

                inputs[n, idx.int():] = mask_value

    return inputs





def skip_add_pyramid(x, seq_len, skip_add="add"):

    if len(x.size()) == 2:

        x = x.unsqueeze(0)

    x_len = x.size()[1] // 2

    even = x[:, torch.arange(0, x_len*2-1, 2).long(), :]

    odd = x[:, torch.arange(1, x_len*2, 2).long(), :]

    if skip_add == "add":

        return (even+odd) / 2, ((seq_len) / 2).int()

    else:

        return even, (seq_len / 2).int()
import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.autograd import Variable





class EncoderRNN(nn.Module):

    def __init__(self, config):

        super(EncoderRNN, self).__init__()

        self.input_size = config["n_channels"]

        self.hidden_size = config["encoder_hidden"]

        self.layers = config.get("encoder_layers", 1)

        self.dnn_layers = config.get("encoder_dnn_layers", 0)

        self.dropout = config.get("encoder_dropout", 0.)

        self.bi = config.get("bidirectional_encoder", False)

        if self.dnn_layers > 0:

            for i in range(self.dnn_layers):

                self.add_module('dnn_' + str(i), nn.Linear(

                    in_features=self.input_size if i == 0 else self.hidden_size,

                    out_features=self.hidden_size

                ))

        gru_input_dim = self.input_size if self.dnn_layers == 0 else self.hidden_size

        self.rnn = nn.GRU(

            gru_input_dim,

            self.hidden_size,

            self.layers,

            dropout=self.dropout,

            bidirectional=self.bi,

            batch_first=True)

        self.gpu = config.get("gpu", False)



    def run_dnn(self, x):

        for i in range(self.dnn_layers):

            x = F.relu(getattr(self, 'dnn_'+str(i))(x))

        return x



    def forward(self, inputs, hidden, input_lengths):

        if self.dnn_layers > 0:

            inputs = self.run_dnn(inputs)

        x = pack_padded_sequence(inputs, input_lengths, batch_first=True)

        output, state = self.rnn(x, hidden)

        output, _ = pad_packed_sequence(output, batch_first=True, padding_value=0.)



        if self.bi:

            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]

        return output, state



    def init_hidden(self, batch_size):

        h0 = Variable(torch.zeros(2 if self.bi else 1, batch_size, self.hidden_size))

        if self.gpu:

            h0 = h0.cuda()

        return h0





class EncoderPyRNN(nn.Module):

    def __init__(self, config):

        super(EncoderPyRNN, self).__init__()

        self.input_size = config["n_channels"]

        self.hidden_size = config["encoder_hidden"]

        self.n_layers = config.get("encoder_layers", 1)

        self.dnn_layers = config.get("encoder_dnn_layers", 0)

        self.dropout = config.get("encoder_dropout", 0.)

        self.bi = config.get("bidirectional_encoder", False)

        self.skip_add = config.get("skip_add_pyramid_encoder", "add")

        self.gpu = config.get("gpu", False)



        if self.dnn_layers > 0:

            for i in range(self.dnn_layers):

                self.add_module('dnn_' + str(i), nn.Linear(

                    in_features=self.input_size if i == 0 else self.hidden_size,

                    out_features=self.hidden_size

                ))

        gru_input_dim = self.input_size if self.dnn_layers == 0 else self.hidden_size



        for i in range(self.n_layers):

            self.add_module('pRNN_' + str(i), nn.GRU(

                input_size=gru_input_dim if i == 0 else self.hidden_size,

                hidden_size=self.hidden_size,

                dropout=self.dropout,

                bidirectional=self.bi,

                batch_first=True))



    def run_dnn(self, x):

        for i in range(self.dnn_layers):

            x = F.relu(getattr(self, 'dnn_'+str(i))(x))

        return x



    def run_pRNN(self, inputs, hidden, input_lengths):

        """

        :param input: (batch, seq_len, input_size)

        :param hidden: (num_layers * num_directions, batch, hidden_size)

        :return:

        """

        for i in range(self.n_layers):

            x = pack_padded_sequence(inputs, input_lengths, batch_first=True)

            output, hidden = getattr(self, 'pRNN_'+str(i))(x, hidden)

            output, _ = pad_packed_sequence(output, batch_first=True, padding_value=0.)

            hidden = hidden



            if self.bi:

                output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]



            if i < self.n_layers - 1:

                inputs, input_lengths = skip_add_pyramid(output, input_lengths, self.skip_add)



        return output, hidden, input_lengths



    def forward(self, inputs, hidden, input_lengths):

        if self.dnn_layers > 0:

            inputs = self.run_dnn(inputs)



        outputs, hidden, input_lengths = self.run_pRNN(inputs, hidden, input_lengths)



        if self.bi:

            hidden = torch.sum(hidden, 0)



        return outputs, hidden, input_lengths



    def init_hidden(self, batch_size):

        h0 = Variable(torch.zeros(2 if self.bi else 1, batch_size, self.hidden_size))

        if self.gpu:

            h0 = h0.cuda()

        return h0
import torch

import torch.nn as nn

import torch.nn.functional as F





class Decoder(nn.Module):

    def __init__(self, config):

        super(Decoder, self).__init__()

        self.batch_size = config["batch_size"]

        self.hidden_size = config["decoder_hidden"]

        embedding_dim = config.get("embedding_dim", None)

        self.embedding_dim = embedding_dim if embedding_dim is not None else self.hidden_size

        self.embedding = nn.Embedding(config.get("n_classes", 32), self.embedding_dim, padding_idx=0)

        self.rnn = nn.GRU(

            input_size=self.embedding_dim+self.hidden_size if config['decoder'].lower() == 'bahdanau' else self.embedding_dim,

            hidden_size=self.hidden_size,

            num_layers=config.get("decoder_layers", 1),

            dropout=config.get("decoder_dropout", 0),

            bidirectional=config.get("bidirectional_decoder", False),

            batch_first=True)

        if config['decoder'] != "RNN":

            self.attention = Attention(

                self.batch_size,

                self.hidden_size,

                method=config.get("attention_score", "dot"),

                mlp=config.get("attention_mlp_pre", False))



        self.gpu = config.get("gpu", False)

        self.decoder_output_fn = F.log_softmax if config.get('loss', 'NLL') == 'NLL' else None



    def forward(self, **kwargs):

        """ Must be overrided """

        raise NotImplementedError





class RNNDecoder(Decoder):

    def __init__(self, config):

        super(RNNDecoder, self).__init__(config)

        self.output_size = config.get("n_classes", 32)

        self.character_distribution = nn.Linear(self.hidden_size, self.output_size)



    def forward(self, **kwargs):

        input = kwargs["input"]

        hidden = kwargs["hidden"]

        # RNN (Eq 7 paper)

        embedded = self.embedding(input).unsqueeze(0)

        rnn_input = torch.cat((embedded, hidden.unsqueeze(0)), 2)  # NOTE : Tf concats `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.

        # rnn_output, rnn_hidden = self.rnn(rnn_input.transpose(1, 0), hidden.unsqueeze(0))

        rnn_output, rnn_hidden = self.rnn(embedded.transpose(1, 0), hidden.unsqueeze(0))

        output = rnn_output.squeeze(1)

        output = self.character_distribution(output)



        if self.decoder_output_fn:

            output = self.decoder_output_fn(output, -1)



        return output, rnn_hidden.squeeze(0)





class BahdanauDecoder(Decoder):

    """

        Corresponds to BahdanauAttnDecoderRNN in Pytorch tuto

    """



    def __init__(self, config):

        super(BahdanauDecoder, self).__init__(config)

        self.output_size = config.get("n_classes", 32)

        self.character_distribution = nn.Linear(self.hidden_size, self.output_size)



    def forward(self, **kwargs):

        """

        :param input: [B]

        :param prev_context: [B, H]

        :param prev_hidden: [B, H]

        :param encoder_outputs: [B, T, H]

        :return: output (B), context (B, H), prev_hidden (B, H), weights (B, T)

        """



        input = kwargs["input"]

        prev_hidden = kwargs["prev_hidden"]

        encoder_outputs = kwargs["encoder_outputs"]

        seq_len = kwargs.get("seq_len", None)



        # check inputs

        assert input.size() == torch.Size([self.batch_size])

        assert prev_hidden.size() == torch.Size([self.batch_size, self.hidden_size])



        # Attention weights

        weights = self.attention.forward(prev_hidden, encoder_outputs, seq_len)  # B x T

        context = weights.unsqueeze(1).bmm(encoder_outputs).squeeze(1)  # [B x H]



        # embed characters

        embedded = self.embedding(input).unsqueeze(0)

        assert embedded.size() == torch.Size([1, self.batch_size, self.embedding_dim])



        rnn_input = torch.cat((embedded, context.unsqueeze(0)), 2)



        outputs, hidden = self.rnn(rnn_input.transpose(1, 0), prev_hidden.unsqueeze(0)) # 1 x B x N, B x N



        # output = self.proj(torch.cat((outputs.squeeze(0), context), 1))

        output = self.character_distribution(outputs.squeeze(0))



        if self.decoder_output_fn:

            output = self.decoder_output_fn(output, -1)



        if len(output.size()) == 3:

            output = output.squeeze(1)



        return output, hidden.squeeze(0), weights





class LuongDecoder(Decoder):

    """

        Corresponds to AttnDecoderRNN

    """



    def __init__(self, config):

        super(LuongDecoder, self).__init__(config)

        self.output_size = config.get("n_classes", 32)

        self.character_distribution = nn.Linear(2*self.hidden_size, self.output_size)



    def forward(self, **kwargs):

        """

        :param input: [B]

        :param prev_context: [B, H]

        :param prev_hidden: [B, H]

        :param encoder_outputs: [B, T, H]

        :return: output (B, V), context (B, H), prev_hidden (B, H), weights (B, T)

        https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py

        TF says : Perform a step of attention-wrapped RNN.

        - Step 1: Mix the `inputs` and previous step's `attention` output via

          `cell_input_fn`.

        - Step 2: Call the wrapped `cell` with this input and its previous state.

        - Step 3: Score the cell's output with `attention_mechanism`.

        - Step 4: Calculate the alignments by passing the score through the

          `normalizer`.

        - Step 5: Calculate the context vector as the inner product between the

          alignments and the attention_mechanism's values (memory).

        - Step 6: Calculate the attention output by concatenating the cell output

          and context through the attention layer (a linear layer with

          `attention_layer_size` outputs).

        Args:

          inputs: (Possibly nested tuple of) Tensor, the input at this time step.

          state: An instance of `AttentionWrapperState` containing

            tensors from the previous time step.

        Returns:

          A tuple `(attention_or_cell_output, next_state)`, where:

          - `attention_or_cell_output` depending on `output_attention`.

          - `next_state` is an instance of `AttentionWrapperState`

             containing the state calculated at this time step.

        Raises:

          TypeError: If `state` is not an instance of `AttentionWrapperState`.

        """

        input = kwargs["input"]

        prev_hidden = kwargs["prev_hidden"]

        encoder_outputs = kwargs["encoder_outputs"]

        seq_len = kwargs.get("seq_len", None)



        # RNN (Eq 7 paper)

        embedded = self.embedding(input).unsqueeze(1) # [B, H]

        prev_hidden = prev_hidden.unsqueeze(0)

        # rnn_input = torch.cat((embedded, prev_context), -1) # NOTE : Tf concats `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.

        # rnn_output, hidden = self.rnn(rnn_input.transpose(1, 0), prev_hidden)

        rnn_output, hidden = self.rnn(embedded, prev_hidden)

        rnn_output = rnn_output.squeeze(1)



        # Attention weights (Eq 6 paper)

        weights = self.attention.forward(rnn_output, encoder_outputs, seq_len) # B x T

        context = weights.unsqueeze(1).bmm(encoder_outputs).squeeze(1)  # [B x N]



        # Projection (Eq 8 paper)

        # /!\ Don't apply tanh on outputs, it fucks everything up

        output = self.character_distribution(torch.cat((rnn_output, context), 1))



        # Apply log softmax if loss is NLL

        if self.decoder_output_fn:

            output = self.decoder_output_fn(output, -1)



        if len(output.size()) == 3:

            output = output.squeeze(1)



        return output, hidden.squeeze(0), weights





class Attention(nn.Module):

    """

    Inputs:

        last_hidden: (batch_size, hidden_size)

        encoder_outputs: (batch_size, max_time, hidden_size)

    Returns:

        attention_weights: (batch_size, max_time)

    """

    def __init__(self, batch_size, hidden_size, method="dot", mlp=False):

        super(Attention, self).__init__()

        self.method = method

        self.hidden_size = hidden_size

        if method == 'dot':

            pass

        elif method == 'general':

            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)

        elif method == "concat":

            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)

            self.va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size))

        elif method == 'bahdanau':

            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)

            self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)

            self.va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size))

        else:

            raise NotImplementedError



        self.mlp = mlp

        if mlp:

            self.phi = nn.Linear(hidden_size, hidden_size, bias=False)

            self.psi = nn.Linear(hidden_size, hidden_size, bias=False)



    def forward(self, last_hidden, encoder_outputs, seq_len=None):

        batch_size, seq_lens, _ = encoder_outputs.size()

        if self.mlp:

            last_hidden = self.phi(last_hidden)

            encoder_outputs = self.psi(encoder_outputs)



        attention_energies = self._score(last_hidden, encoder_outputs, self.method)

        # attn_energies = Variable(torch.zeros(batch_size, seq_lens))  # B x S



        if seq_len is not None:

            attention_energies = mask_3d(attention_energies, seq_len, -float('inf'))



        return F.softmax(attention_energies, -1)



    def _score(self, last_hidden, encoder_outputs, method):

        """

        Computes an attention score

        :param last_hidden: (batch_size, hidden_dim)

        :param encoder_outputs: (batch_size, max_time, hidden_dim)

        :param method: str (`dot`, `general`, `concat`)

        :return:

        """



        # assert last_hidden.size() == torch.Size([batch_size, self.hidden_size]), last_hidden.size()

        assert encoder_outputs.size()[-1] == self.hidden_size



        if method == 'dot':

            last_hidden = last_hidden.unsqueeze(-1)

            return encoder_outputs.bmm(last_hidden).squeeze(-1)



        elif method == 'general':

            x = self.Wa(last_hidden)

            x = x.unsqueeze(-1)

            return encoder_outputs.bmm(x).squeeze(-1)



        elif method == "concat":

            x = last_hidden.unsqueeze(1)

            x = F.tanh(self.Wa(torch.cat((x, encoder_outputs), 1)))

            return x.bmm(self.va.unsqueeze(2)).squeeze(-1)



        elif method == "bahdanau":

            x = last_hidden.unsqueeze(1)

            out = F.tanh(self.Wa(x) + self.Ua(encoder_outputs))

            return out.bmm(self.va.unsqueeze(2)).squeeze(-1)



        else:

            raise NotImplementedError



    def extra_repr(self):

        return 'score={}, mlp_preprocessing={}'.format(

            self.method, self.mlp)
import torch

import torch.nn as nn

import torch.nn.functional as F

import random



from torch.autograd import Variable





class Seq2Seq(nn.Module):

    """

        Sequence to sequence module

    """



    def __init__(self, config):

        super(Seq2Seq, self).__init__()

        self.SOS = config.get("start_index", 1),

        self.vocab_size = config.get("n_classes", 32)

        self.batch_size = config.get("batch_size", 1)

        self.sampling_prob = config.get("sampling_prob", 0.)

        self.gpu = config.get("gpu", False)



        # Encoder

        if config["encoder"] == "PyRNN":

            self._encoder_style = "PyRNN"

            self.encoder = EncoderPyRNN(config)

        else:

            self._encoder_style = "RNN"

            self.encoder = EncoderRNN(config)



        # Decoder

        self.use_attention = config["decoder"] != "RNN"

        if config["decoder"] == "Luong":

            self.decoder = LuongDecoder(config)

        elif config["decoder"] == "Bahdanau":

            self.decoder = BahdanauDecoder(config)

        else:

            self.decoder = RNNDecoder(config)



        if config.get('loss') == 'cross_entropy':

            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

            config['loss'] = 'cross_entropy'

        else:

            self.loss_fn = torch.nn.NLLLoss(ignore_index=0)

            config['loss'] = 'NLL'

        self.loss_type = config['loss']

        print(config)



    def encode(self, x, x_len):



        batch_size = x.size()[0]

        init_state = self.encoder.init_hidden(batch_size)

        if self._encoder_style == "PyRNN":

            encoder_outputs, encoder_state, input_lengths = self.encoder.forward(x, init_state, x_len)

        else:

            encoder_outputs, encoder_state = self.encoder.forward(x, init_state, x_len)



        assert encoder_outputs.size()[0] == self.batch_size, encoder_outputs.size()

        assert encoder_outputs.size()[-1] == self.decoder.hidden_size



        if self._encoder_style == "PyRNN":

            return encoder_outputs, encoder_state.squeeze(0), input_lengths

        return encoder_outputs, encoder_state.squeeze(0)



    def decode(self, encoder_outputs, encoder_hidden, targets, targets_lengths, input_lengths):

        """

        Args:

            encoder_outputs: (B, T, H)

            encoder_hidden: (B, H)

            targets: (B, L)

            targets_lengths: (B)

            input_lengths: (B)

        Vars:

            decoder_input: (B)

            decoder_context: (B, H)

            hidden_state: (B, H)

            attention_weights: (B, T)

        Outputs:

            alignments: (L, T, B)

            logits: (B*L, V)

            labels: (B*L)

        """



        batch_size = encoder_outputs.size()[0]

        max_length = targets.size()[1]

        # decoder_attns = torch.zeros(batch_size, MAX_LENGTH, MAX_LENGTH)

        decoder_input = Variable(torch.LongTensor([self.SOS] * batch_size)).squeeze(-1)

        decoder_context = encoder_outputs.transpose(1, 0)[-1]

        decoder_hidden = encoder_hidden



        alignments = Variable(torch.zeros(max_length, encoder_outputs.size(1), batch_size))

        logits = Variable(torch.zeros(max_length, batch_size, self.decoder.output_size))



        if self.gpu:

            decoder_input = decoder_input.cuda()

            decoder_context = decoder_context.cuda()

            logits = logits.cuda()



        for t in range(max_length):



            # The decoder accepts, at each time step t :

            # - an input, [B]

            # - a context, [B, H]

            # - an hidden state, [B, H]

            # - encoder outputs, [B, T, H]



            check_size(decoder_input, self.batch_size)

            check_size(decoder_hidden, self.batch_size, self.decoder.hidden_size)



            # The decoder outputs, at each time step t :

            # - an output, [B]

            # - a context, [B, H]

            # - an hidden state, [B, H]

            # - weights, [B, T]



            if self.use_attention:

                check_size(decoder_context, self.batch_size, self.decoder.hidden_size)

                outputs, decoder_hidden, attention_weights = self.decoder.forward(

                    input=decoder_input.long(),

                    prev_hidden=decoder_hidden,

                    encoder_outputs=encoder_outputs,

                    seq_len=input_lengths)

                alignments[t] = attention_weights.transpose(1, 0)

            else:

                outputs, hidden = self.decoder.forward(

                    input=decoder_input.long(),

                    hidden=decoder_hidden)



            # print(outputs[0])

            logits[t] = outputs



            use_teacher_forcing = random.random() > self.sampling_prob



            if use_teacher_forcing and self.training:

                decoder_input = targets[:, t]



            # SCHEDULED SAMPLING

            # We use the target sequence at each time step which we feed in the decoder

            else:

                # TODO Instead of taking the direct one-hot prediction from the previous time step as the original paper

                # does, we thought it is better to feed the distribution vector as it encodes more information about

                # prediction from previous step and could reduce bias.

                topv, topi = outputs.data.topk(1)

                decoder_input = topi.squeeze(-1).detach()





        labels = targets.contiguous().view(-1)



        if self.loss_type == 'NLL': # ie softmax already on outputs

            mask_value = -float('inf')

            print(torch.sum(logits, dim=2))

        else:

            mask_value = 0



        logits = mask_3d(logits.transpose(1, 0), targets_lengths, mask_value)

        logits = logits.contiguous().view(-1, self.vocab_size)



        return logits, labels.long(), alignments



    @staticmethod

    def custom_loss(logits, labels):



        # create a mask by filtering out all tokens that ARE NOT the padding token

        tag_pad_token = 0

        mask = (labels > tag_pad_token).float()



        # count how many tokens we have

        nb_tokens = int(torch.sum(mask).data[0])



        # pick the values for the label and zero out the rest with the mask

        logits = logits[range(logits.shape[0]), labels] * mask



        # compute cross entropy loss which ignores all <PAD> tokens

        ce_loss = -torch.sum(logits) / nb_tokens



        return ce_loss



    def step(self, batch):

        x, y, x_len, y_len = batch

        if self.gpu:

            x = x.cuda()

            y = y.cuda()

            x_len = x_len.cuda()

            y_len = y_len.cuda()



        if self._encoder_style == "PyRNN":

            encoder_out, encoder_state, x_len = self.encode(x, x_len)

        else:

            encoder_out, encoder_state = self.encode(x, x_len)

        logits, labels, alignments = self.decode(encoder_out, encoder_state, y, y_len, x_len)

        return logits, labels, alignments



    def loss(self, batch):

        logits, labels, alignments = self.step(batch)

        loss = self.loss_fn(logits, labels)

        # loss2 = self.custom_loss(logits, labels)

        return loss, logits, labels, alignments
def train(model, optimizer, train_loader, state):

    epoch, n_epochs, train_steps = state



    losses = []

    cers = []



    # t = tqdm.tqdm(total=min(len(train_loader), train_steps))

    t = tqdm.tqdm(train_loader)

    model.train()



    for batch in t:

        t.set_description("Epoch {:.0f}/{:.0f} (train={})".format((epoch+1), n_epochs, model.training))

        loss, _, _, _ = model.loss(batch)

        losses.append(loss.item())

        # Reset gradients

        optimizer.zero_grad()

        # Compute gradients

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)

        optimizer.step()

        t.set_postfix(loss='{:05.3f}'.format(loss.item()), avg_loss='{:05.3f}'.format(np.mean(losses)))

        t.update()



    return model, optimizer

    # print(" End of training:  loss={:05.3f} , cer={:03.1f}".format(np.mean(losses), np.mean(cers)*100))
def visualize_train_test(train_vals,test_vals,x_label,y_label):

    %matplotlib inline

    epoch_count = [i for i in range(len(train_vals))]

    plt.plot(epoch_count, train_vals, 'r--')

    plt.plot(epoch_count, test_vals, 'b-')

    plt.legend(['Training '+y_label, 'Test '+y_label])

    plt.xlabel(x_label)

    plt.ylabel(y_label)

    plt.show()
visualize_train_test([1.,2.,3.,4.],[2.,1.,3.,4.],"epoch","accuacy")
def showAttention(input_sentence, output_words, gold_words,attentions):

    # Set up figure with colorbar

    fig = plt.figure()

    ax = fig.add_subplot(111)

    cax = ax.matshow(attentions, cmap='bone')

    fig.colorbar(cax)

    is_match = (output_words==gold_words)



    # Set up axes

    ax.set_xticklabels([''] + input_sentence.split(' ') +

                       ['<EOS>'], rotation=90)

    ax.set_yticklabels([''] + output_words)



    # Show label at every tick

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))



    plt.figtext(0.5, 0.01, str(is_match)+" match", wrap=True, horizontalalignment='center', fontsize=12)

    plt.show()
def print_outputs_and_show_attention(alignments,labels,batch,preds,data_loader):

    input_sequence = []

    char2int_dict = data_loader.dataset.char2int

    int2char_dict = {v: k for k, v in char2int_dict.items()}

    chars = list(int2char_dict.values())

    ix = random.randint(0,data_loader.batch_size-1) 

    align = alignments.detach().cpu().numpy()[:, :, ix]

    input_sequence = []

    for one_hot in batch[0][ix]:

        i, = np.where(one_hot == 1)

        if len(i)>0:

            input_sequence.append(chars[i[0]])

    gold_sequence = [int2char_dict[int(i)] for i in preds[ix*15:ix*15+15] if int(i) in int2char_dict.keys()]

    pred_sequence = [int2char_dict[int(i)] for i in labels[ix*15:ix*15+15] if int(i) in int2char_dict.keys()]

    print("input:\t"," ".join(input_sequence)," (",len(input_sequence),")")

    print("gold:\t"," ".join(gold_sequence)," (",len(gold_sequence),")")

    print("prediction:\t"," ".join(pred_sequence),"(",len(pred_sequence),")")

    print("match:\t",(gold_sequence==pred_sequence),"\n")

    showAttention(" ".join(input_sequence),pred_sequence,gold_sequence,align)
def evaluate(model, eval_loader,visualize=False):



    losses = []

    accs = []



    t = tqdm.tqdm(eval_loader)

    model.eval()



    with torch.no_grad():

        for batch in t:

            t.set_description(" Evaluating... (train={})".format(model.training))

            loss, logits, labels, alignments = model.loss(batch)

            preds = logits.detach().cpu().numpy()

            # acc = np.sum(np.argmax(preds, -1) == labels.detach().cpu().numpy()) / len(preds)

            acc = 100 - 100 * editdistance.eval(np.argmax(preds, -1), labels.detach().cpu().numpy()) / len(preds)

            losses.append(loss.item())

            accs.append(acc)

            t.set_postfix(avg_acc='{:05.3f}'.format(np.mean(accs)), avg_loss='{:05.3f}'.format(np.mean(losses)))

            t.update()

            if visualize:

                print_outputs_and_show_attention(alignments,labels,batch,preds,eval_loader)

    print("  End of evaluation : loss {:05.3f} , acc {:03.1f}".format(np.mean(losses), np.mean(accs)))

    return {'loss': np.mean(losses), 'acc': np.mean(accs)}
def evaluate_random(model,batch_size,num=1):

    eval_dataset = ToyDataset(5, 15, type='eval')

    eval_loader = data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate,drop_last=True)

    char2int_dict = eval_loader.dataset.char2int

    int2char_dict = {v: k for k, v in char2int_dict.items()}

    chars = list(int2char_dict.values())

    model.eval()

    counter=0

    with torch.no_grad():

        for batch in eval_loader:

            if counter>=num:

                break

            loss, logits, labels, alignments = model.loss(batch)

            preds = logits.detach().cpu().numpy()

            preds = np.argmax(preds, -1)

            print_outputs_and_show_attention(alignments,labels,batch,preds,eval_loader)

            counter=counter+1
import argparse

import torch

import json

import os



from torch.utils import data





def run(num_epochs=FLAGS.epochs,batch_size=30):

    USE_CUDA = torch.cuda.is_available()



    config_path = os.path.join("../input/", FLAGS.config)



    if not os.path.exists(config_path):

        raise FileNotFoundError



    with open(config_path, "r") as f:

        config = json.load(f)



    config["gpu"] = torch.cuda.is_available()



    dataset = ToyDataset(5, 15)

    eval_dataset = ToyDataset(5, 15, type='eval')

    train_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate, drop_last=True)

    eval_loader = data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate,

                                  drop_last=True)

    config["batch_size"] = batch_size





    # Models

    model = Seq2Seq(config)



    if USE_CUDA:

        model = model.cuda()



    # Optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", .001))



    print("=" * 60)

    print(model)

    print("=" * 60)

    for k, v in sorted(config.items(), key=lambda i: i[0]):

        print(" (" + k + ") : " + str(v))

    print()

    print("=" * 60)



    print("\nInitializing weights...")

    for name, param in model.named_parameters():

        if 'bias' in name:

            torch.nn.init.constant_(param, 0.0)

        elif 'weight' in name:

            torch.nn.init.xavier_normal_(param)





    train_results = []

    eval_results = []

    for epoch in range(num_epochs):

        run_state = (epoch, num_epochs, FLAGS.train_size)



        # Train needs to return model and optimizer, otherwise the model keeps restarting from zero at every epoch

        model, optimizer = train(model, optimizer, train_loader, run_state)

        eval_res = evaluate(model, eval_loader)

        train_res = evaluate(model,train_loader)

        train_results.append(train_res)

        eval_results.append(eval_res)

        

    return model,optimizer,config,train_results,eval_results

model,optimizer,config,train_results,eval_results = run(num_epochs=10)
train_losses = [train_res["loss"] for train_res in train_results]

eval_losses = [eval_res["loss"] for eval_res in eval_results]

visualize_train_test(train_losses,eval_losses,"epoch","loss")
train_accs = [train_res["acc"] for train_res in train_results]

eval_accs = [eval_res["acc"] for eval_res in eval_results]

visualize_train_test(train_accs,eval_accs,"epoch","accuracy")
# Print model's state_dict

print("Model's state_dict:")

for param_tensor in model.state_dict():

    print(param_tensor, "\t", model.state_dict()[param_tensor].size())



# Print optimizer's state_dict

print("Optimizer's state_dict:")

for var_name in optimizer.state_dict():

    print(var_name, "\t", optimizer.state_dict()[var_name])
!mkdir -p "model"
model_path = "model/my.model"
torch.save(model.state_dict(), model_path)
model_path = "model/my.model"
config_path = os.path.join("../input/", FLAGS.config)



if not os.path.exists(config_path):

    raise FileNotFoundError



with open(config_path, "r") as f:

    config = json.load(f)

config["batch_size"]=5

model = Seq2Seq(config)

model.load_state_dict(torch.load(model_path))

model
test_dataset = ToyDataset(5, 15, type='eval')

test_loader = data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=pad_collate,drop_last=True)
len(test_loader.dataset.set),config["batch_size"]
model.eval()

evaluate(model,test_loader)
evaluate_random(model,config["batch_size"])