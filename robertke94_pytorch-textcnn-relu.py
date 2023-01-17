# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import nltk
import time
from datetime import timedelta
import numpy as np
from sklearn import metrics


def save_model(model, model_path):
    """Save model."""
    torch.save(model.state_dict(), model_path)


def load_model(model, model_path, use_cuda=False):
    """Load model."""
    map_location = 'cpu'
    if use_cuda and torch.cuda.is_available():
        map_location = 'cuda:0'
    model.load_state_dict(torch.load(model_path, map_location))
    return model
class TextCNN(nn.Module):
    """Text classification model by character CNN, the implementation of paper
    'Yoon Kim. 2014. Convolution Neural Networks for Sentence
    Classification.'
    """

    def __init__(self, args):
        super(TextCNN, self).__init__()

        vocab_size = args["vocab_size"]
        pretrained_embed = args["pretrained_embed"]
        padding_idx = args["padding_idx"]
        num_classes = 1
        kernel_nums = [100, 100, 100]
        kernel_sizes = [3, 4, 5]
        embed_dim = 300
        hidden_dim = 100
        drop_prob = 0.5

        # no support for pre-trained embedding currently
        if pretrained_embed is None:
            self.embed = nn.Embedding(vocab_size, embed_dim)
        else:
            self.embed = nn.Embedding.from_pretrained(
                pretrained_embed, freeze=False)
        self.embed.padding_idx = padding_idx
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, kn, ks)
             for kn, ks in zip(kernel_nums, kernel_sizes)])
        self.fc = nn.Linear(sum(kernel_nums), hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop_prob)
        self.out = nn.Linear(hidden_dim, num_classes)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, word_seq):
        # embed
        e = self.drop(self.embed(word_seq))  # [b,msl]->[b,msl,e]

        # conv and pool, [b,msl,e]->[b,h,msl]
        e = e.transpose(1, 2)  # [b,msl,e]->[b,e,msl]
        ps = []
        for conv in self.convs:
            c = conv(e)  # [b,e,msl]->[b,h,msl-k]
            p = F.max_pool1d(c, kernel_size=c.size(-1)).squeeze(-1)  # [b,h]
            ps.append(p)
        p = torch.cat(ps, dim=1)  # [b,h]

        # feed-forward, [b,h]->[b]
        f = self.drop(self.act(self.fc(p)))
        logits = self.out(f).squeeze(-1)

        return logits


class Trainer(object):
    """Trainer."""

    def __init__(self, **kwargs):
        self.n_epochs = kwargs["epochs"]
        self.batch_size = kwargs["batch_size"]
        self.validate = kwargs["validate"]
        self.save_best_dev = kwargs["save_best_dev"]
        self.use_cuda = kwargs["use_cuda"]
        self.print_every_step = kwargs["print_every_step"]
        self.optimizer = kwargs["optimizer"]
        self.model_path = kwargs["model_path"]
        self.eval_metrics = kwargs["eval_metrics"]

        self._best_accuracy = 0.0

        self.device = 'cpu'
        if torch.cuda.is_available() and self.use_cuda:
            self.device = 'cuda:0'

    def train(self, network, train_data, dev_data=None):
        # transfer model to gpu if available
        network = network.to(self.device)

        # define batch iterator
        train_iter = torchtext.data.Iterator(
            dataset=train_data, batch_size=self.batch_size,
            train=True, shuffle=True, sort=False,
            device=self.device)

        # define Tester over dev data
        if self.validate:
            default_valid_args = {
                "batch_size": max(8, self.batch_size // 10),
                "use_cuda": self.use_cuda}
            validator = Tester(**default_valid_args)

        start = time.time()
        for epoch in range(1, self.n_epochs + 1):
            # turn on network training mode
            network.train()

            # initialize iterator
            train_iter.init_epoch()

            # one forward and backward pass
            self._train_step(
                train_iter, network, start=start,
                n_print=self.print_every_step, epoch=epoch)

            # validation
            if self.validate:
                if dev_data is None:
                    raise RuntimeError(
                        "self.validate is True in trainer, "
                        "but dev_data is None."
                        " Please provide the validation data.")
                eval_results = validator.test(network, dev_data)

                if self.save_best_dev and self.best_eval_result(eval_results):
                    save_model(network, self.model_path)
                    print("Saved better model selected by validation.")

    def _train_step(self, data_iterator, network, **kwargs):
        """Training process in one epoch.
        """
        step = 0
        for batch in data_iterator:
            text, target = batch.text, batch.target

            self.optimizer.zero_grad()
            logits = network(text)
            loss = network.loss(logits, target.float())
            loss.backward()
            self.optimizer.step()

            if kwargs["n_print"] > 0 and step % kwargs["n_print"] == 0:
                end = time.time()
                diff = timedelta(seconds=round(end - kwargs["start"]))
                print_output = "[epoch: {:>3} step: {:>4}]" \
                    " train loss: {:>4.6} time: {}".format(
                        kwargs["epoch"], step, loss.item(), diff)
                print(print_output)

            step += 1

    def best_eval_result(self, eval_results):
        """Check if the current epoch yields better validation results.

        :param eval_results: dict, format {metrics_name: value}
        :return: bool, True means current results on dev set is the best.
        """
        assert self.eval_metrics in eval_results, \
            "Evaluation doesn't contain metrics '{}'." \
            .format(self.eval_metrics)

        accuracy = eval_results[self.eval_metrics]
        if accuracy > self._best_accuracy:
            self._best_accuracy = accuracy
            return True
        else:
            return False


class Tester(object):
    """Tester."""

    def __init__(self, **kwargs):
        self.batch_size = kwargs["batch_size"]
        self.use_cuda = kwargs["use_cuda"]
        self.device = 'cpu'
        if torch.cuda.is_available() and self.use_cuda:
            self.device = 'cuda:0'

    def test(self, network, dev_data, threshold=0.33):
        # transfer model to gpu if available
        network = network.to(self.device)

        # turn on the testing mode; clean up the history
        network.eval()
        output_list = []
        truth_list = []

        # define batch iterator
        data_iter = torchtext.data.Iterator(
            dataset=dev_data, batch_size=self.batch_size,
            train=False, device=self.device, sort=False)

        # predict
        for batch in data_iter:
            text, target = batch.text, batch.target

            with torch.no_grad():
                prediction = network(text)

            output_list.append(prediction.detach())
            truth_list.append(target.detach())

        # evaluate
        eval_results = self.evaluate(output_list, truth_list, threshold)
        print("[tester] {}".format(self.print_eval_results(eval_results)))

        return eval_results

    def evaluate(self, predict, truth, threshold=0.33):
        """Compute evaluation metrics.

        :param predict: list of Tensor
        :param truth: list of dict
        :param threshold: threshold of positive probability
        :return eval_results: dict, format {name: metrics}.
        """
        y_trues, y_preds = [], []
        for y_true, logit in zip(truth, predict):
            y_pred = (torch.sigmoid(logit) > threshold).long().cpu().numpy()
            y_true = y_true.cpu().numpy()
            y_trues.append(y_true)
            y_preds.append(y_pred)
        y_true = np.concatenate(y_trues, axis=0)
        y_pred = np.concatenate(y_preds, axis=0)

        precision = metrics.precision_score(y_true, y_pred, pos_label=1)
        recall = metrics.recall_score(y_true, y_pred, pos_label=1)
        f1 = metrics.f1_score(y_true, y_pred, pos_label=1)

        metrics_dict = {"precision": precision, "recall": recall, "f1": f1}

        return metrics_dict

    def print_eval_results(self, results):
        """Override this method to support more print formats.
        :param results: dict, (str: float) is (metrics name: value)
        """
        return ", ".join(
            [str(key) + "=" + "{:.4f}".format(value)
             for key, value in results.items()])


class Predictor(object):
    """An interface for predicting outputs based on trained models.
    """

    def __init__(self, batch_size=8, use_cuda=False):
        self.batch_size = batch_size
        self.use_cuda = use_cuda

        self.device = 'cpu'
        if torch.cuda.is_available() and self.use_cuda:
            self.device = 'cuda:0'

    def predict(self, network, data, threshold=0.33):
        # transfer model to gpu if available
        network = network.to(self.device)

        # turn on the testing mode; clean up the history
        network.eval()
        batch_output = []

        # define batch iterator
        data_iter = torchtext.data.Iterator(
            dataset=data, batch_size=self.batch_size,
            train=False, device=self.device, sort=False)

        for batch in data_iter:
            text = batch.text

            with torch.no_grad():
                prediction = network(text)

            batch_output.append(prediction.detach())

        return self._post_processor(batch_output, threshold)

    def _post_processor(self, batch_output, threshold=0.33):
        """Convert logit tensor to label."""
        y_preds = []
        for logit in batch_output:
            y_pred = (torch.sigmoid(logit) > threshold).long().cpu().numpy()
            y_preds.append(y_pred)
        y_pred = np.concatenate(y_preds, axis=0)

        return y_pred
train_path = '../input/train.csv'
test_path = '../input/test.csv'
embed_path = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
submission_path = './submission.csv'
model_path = './default_model.pkl'
def pre():
    """Pre-process model."""

    print("Pre-processing...")

    # load data
    fix_length = 100
    text = torchtext.data.Field(
        sequential=True, use_vocab=True, lower=True,
        tokenize=nltk.word_tokenize, batch_first=True,
        is_target=False, fix_length=fix_length)
    target = torchtext.data.Field(
        sequential=False, use_vocab=False,
        batch_first=True, is_target=True)
    dataset = torchtext.data.TabularDataset(
        train_path, format='csv',
        fields={"question_text": ('text', text),
                "target": ('target', target)})
    data_test = torchtext.data.TabularDataset(
        test_path, format='csv',
        fields={"question_text": ('text', text)})

    # build vocab
    text.build_vocab(dataset, data_test, min_freq=3)
    text.vocab.load_vectors(torchtext.vocab.Vectors(embed_path))
    vocab_size = len(text.vocab.itos)
    padding_idx = text.vocab.stoi[text.pad_token]

    # split data
    data_train, data_val = dataset.split(split_ratio=0.9)

    print("train set size:", len(data_train))
    print("val set size:", len(data_val))
    print("test set size:", len(data_test))
    print("vocab size:", len(text.vocab.itos))
    print("embed shape:", text.vocab.vectors.shape)
    print('')

    # # save data
    # save_pickle(data_train, pickle_path, 'data_train.pkl')
    # save_pickle(data_val, pickle_path, 'data_val.pkl')
    # print('')

    args_dict = {
        "data_train": data_train, "data_val": data_val,
        "data_test": data_test, "vocab_size": vocab_size,
        "padding_idx": padding_idx}

    return args_dict
args = pre()
def train(**args):
    """Train model.
    """

    print("Training...")

    # load data and embed
    data_train = args["data_train"]
    pretrained_embed = data_train.fields["text"].vocab.vectors

    # define model
    model_args = {
        "vocab_size": args["vocab_size"],
        "padding_idx": args["padding_idx"],
        "pretrained_embed": pretrained_embed,
    }
    model = TextCNN(model_args)

    # define trainer
    trainer_args = {
        "epochs": 8,
        "batch_size": 128,
        "validate": True,
        "save_best_dev": True,
        "use_cuda": True,
        "print_every_step": 1000,
        "optimizer": torch.optim.Adam(model.parameters(), lr=1e-3),
        "model_path": model_path,
        "eval_metrics": "f1",
    }
    trainer = Trainer(**trainer_args)

    # train
    data_val = args["data_val"]
    trainer.train(model, data_train, dev_data=data_val)

    print('')
train(**args)
def test(**args):
    """Train model.
    """

    print("Testing...")

    # define model
    model_args = {
        "vocab_size": args["vocab_size"],
        "padding_idx": args["padding_idx"],
        "pretrained_embed": None,
    }
    model = TextCNN(model_args)
    load_model(model, model_path, use_cuda=True)

    # define tester
    tester_args = {
        "batch_size": 128,
        "use_cuda": True,
    }
    tester = Tester(**tester_args)

    # test and threshold selection
    data_val = args["data_val"]
    best_thresh, best_f1 = 0., 0.
    for thresh in np.arange(0.1, 0.501, 0.01):
        thresh = np.round(thresh, 2)
        f1 = tester.test(model, data_val, threshold=thresh)["f1"]
        print("threshold: {:>.2f} f1: {:>.4f}".format(thresh, f1))
        if f1 > best_f1:
            best_thresh, best_f1 = thresh, f1

    args["threshold"] = best_thresh

    print("best f1 on dev: {:>.4f} threshold: {:>.2f}".format(
        best_f1, best_thresh))
    print('')

    return args
    args = test(**args)
def infer(**args):
    """Inference using model.
    """

    print("Predicting...")

    # define model
    model_args = {
        "vocab_size": args["vocab_size"],
        "padding_idx": args["padding_idx"],
        "pretrained_embed": None,
    }
    model = TextCNN(model_args)
    load_model(model, model_path, use_cuda=True)

    # define predictor
    predictor = Predictor(batch_size=128, use_cuda=False)

    # predict
    data_test = args["data_test"]
    threshold = args["threshold"]
    y_pred = predictor.predict(model, data_test, threshold=threshold)

    # submit result
    test_df = pd.read_csv(test_path, index_col=False, header=0)
    data = {"qid": test_df["qid"], "prediction": y_pred}
    subm_df = pd.DataFrame(data=data)
    subm_df.to_csv(submission_path, header=True, index=False)

    print("submission saved as {}.".format(submission_path))
    print('')
infer(**args)