import os

import random

import warnings

from typing import Dict, List, Tuple, Union



import numpy as np

import pandas as pd

import tokenizers

import torch

import torch.optim as optim

from sklearn.model_selection import StratifiedKFold

from torch import nn

from torch.utils.data import Dataset



with warnings.catch_warnings():

    warnings.filterwarnings("ignore")

    from transformers import RobertaModel, RobertaTokenizer
RUN_LOCAL = os.environ.get("PWD") != "/kaggle/working"

RUN_LOCAL
SEED = 42



if RUN_LOCAL:

    NUM_EPOCHS = 1

    HOME_DIR = "/workspaces/kaggle"

    BATCH_SIZE = 1

    SKF = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)

else:

    NUM_EPOCHS = 3

    HOME_DIR = "/kaggle"

    BATCH_SIZE = 32

    SKF = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
def seed_everything(seed_value: int) -> None:

    """

    Make notebook results exactly reproducible.



    Parameters

    ----------

    seed_value

        Random seed.

    """

    random.seed(seed_value)

    np.random.seed(seed_value)

    torch.manual_seed(seed_value)

    os.environ["PYTHONHASHSEED"] = str(seed_value)



    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = True





seed_everything(SEED)
def _get_input_data(

    row: pd.Series,

    tokenizer: tokenizers.ByteLevelBPETokenizer,

    max_len: int,

    bos_token_id: int,

    eos_token_id: int,

) -> Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:

    """

    Encode input row so it can be processed by Roberta.



    Encoding follows: <s> sentiment </s></s> encoding </s>



    Parameters

    ----------

    row :

        Raw data

    tokenizer :

        Converts text to stream of tokens

    max_len :

        Maximum length of encoded streams

    bos_token_id :

        Beginning of sentence token

    eos_token_id :

        End of sentence token



    Returns

    -------

    ids :

        Encoded stream of text

    masks :

        1 for tokens corresponding to <s> sentiment </s></s> encoding </s>,

        0 for tokens corresponding to padding

    tweet :

        Raw tweet

    offsets :

        Positions of raw text corresponding to each token

    """

    tweet = " " + " ".join(row.text.lower().split())

    encoding = tokenizer.encode(tweet)

    sentiment_id = tokenizer.encode(row.sentiment).ids

    ids = (

        [bos_token_id]

        + sentiment_id

        + [eos_token_id, eos_token_id]

        + encoding.ids

        + [eos_token_id]

    )

    offsets = [(0, 0)] * 4 + encoding.offsets + [(0, 0)]



    pad_len = max_len - len(ids)

    if pad_len > 0:

        ids += [1] * pad_len

        offsets += [(0, 0)] * pad_len



    ids = torch.tensor(ids)

    masks = torch.where(ids != 1, torch.tensor(1), torch.tensor(0))

    offsets = torch.tensor(offsets)



    assert ids.shape == torch.Size([96])

    assert masks.shape == torch.Size([96])

    assert offsets.shape == torch.Size([96, 2])



    return ids, masks, tweet, offsets
def _get_target_idx(row: pd.Series, tweet: str, offsets: torch.Tensor) -> Tuple[int, int]:

    """

    Get the start and end tokens corresponding to the target.



    Parameters

    ----------

    row

        Raw data

    tweet

        Lowercase tweet with leading space

    offsets

        Positions of raw text corresponding to each token



    Returns

    -------

    int

        First token to contain a character from selected text

    int

        Last token to contain a character from selected text

    """

    assert offsets.shape == torch.Size([96, 2])



    selected_text = " " + " ".join(row.selected_text.lower().split())



    len_st = len(selected_text) - 1

    idx0 = None

    idx1 = None



    # First the starting and ending index of the selected text within the tweet.

    for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):

        if " " + tweet[ind : ind + len_st] == selected_text:

            idx0 = ind

            idx1 = ind + len_st

            break



    # Character targets: 0 outside of selected text, 1 inside it.

    char_targets = [0] * len(tweet)

    if idx0 is not None and idx1 is not None:

        for ct in range(idx0, idx1):

            char_targets[ct] = 1



    # If any token contains any character from the selected text, then set

    # the target value for that token to be 1.

    target_idx: List[int] = []

    for j, (offset1, offset2) in enumerate(offsets):

        left = offset1.item()

        right = offset2.item()

        assert isinstance(left, int)

        assert isinstance(right, int)

        if sum(char_targets[left:right]) > 0:

            target_idx.append(j)



    start_idx = target_idx[0]

    end_idx = target_idx[-1]



    return start_idx, end_idx
def _process_row(

    row: pd.Series,

    tokenizer: tokenizers.ByteLevelBPETokenizer,

    max_len: int,

    bos_token_id: int,

    eos_token_id: int,

) -> Dict[str, Union[torch.Tensor, str, int]]:

    """

    Process row so it can be fed into Roberta.



    Processed row contains:

    - ids (tokenised input)

    - masks (1 outside padding)

    - tweet (lowercase tweet with leading whitespace)

    - offsets (mapping of tokens to tweet)



    If training, it also contains:

    - start_idx (first token containing part of selected text)

    - end_idx (last token containing part of selected text)



    Parameters

    ----------

    row :

        Raw data

    tokenizer :

        Converts text to stream of tokens

    max_len :

        Maximum length of encoded streams

    bos_token_id :

        Beginning of sentence token

    eos_token_id :

        End of sentence token



    Returns

    -------

    Dictionary that can be processed by training loop.

    """

    data: Dict[str, Union[torch.Tensor, str, int]] = {}



    ids, masks, tweet, offsets = _get_input_data(

        row, tokenizer, max_len, bos_token_id, eos_token_id

    )

    data.update({"ids": ids})

    data["masks"] = masks

    data["tweet"] = tweet

    data["offsets"] = offsets



    if "selected_text" in row:

        start_idx, end_idx = _get_target_idx(row, tweet, offsets)

        data["start_idx"] = start_idx

        data["end_idx"] = end_idx



    return data
class TweetDataset(Dataset):

    """

    Processed rows that can be used by PyTorch for training / inference.



    Processed rows contain:

    - ids (tokenised input)

    - masks (1 outside padding)

    - tweet (lowercase tweet with leading whitespace)

    - offsets (mapping of tokens to tweet)



    If training, they also contain:

    - start_idx (first token containing part of selected text)

    - end_idx (last token containing part of selected text)

    """



    def __init__(self, df: pd.DataFrame, max_len: int = 96) -> None:

        """

        Initialise TweetDataset.



        Parameters

        ----------

        df

            Raw DataFrame (e.g. training, validation, testing).

        max_len

            Maximum number of tokens with which to encode tweets.

        """

        self.df = df

        self.max_len = max_len

        self.labeled = "selected_text" in df



        tokenizer = RobertaTokenizer.from_pretrained(f"{HOME_DIR}/input/roberta-base")

        tokenizer.save_vocabulary(".")

        self.bos_token_id = tokenizer.bos_token_id

        self.eos_token_id = tokenizer.eos_token_id



        # vocab.json and merges.txt come from save_vocabulary above

        self.tokenizer = tokenizers.ByteLevelBPETokenizer(

            vocab_file="./vocab.json",

            merges_file="./merges.txt",

            lowercase=True,

            add_prefix_space=True,

        )



    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str, int]]:

        """

        Return encoded version of index-th element of dataset.



        Parameters

        ----------

        index

            Which element of the dataset to get.



        Returns

        -------

        dict

            Processed row of data.

        """

        row = self.df.iloc[index]

        data = _process_row(

            row, self.tokenizer, self.max_len, self.bos_token_id, self.eos_token_id

        )

        return data



    def __len__(self) -> int:

        """

        Get number of rows in dataset.



        Returns

        -------

        int

        """

        return len(self.df)





def get_train_val_loaders(

    df: pd.DataFrame, train_idx: np.array, val_idx: np.array, batch_size: int = 8

) -> Dict[str, torch.utils.data.DataLoader]:

    """

    Get train and validation data loaders.



    Parameters

    ----------

    df

        Entire training dataset (will be split into train and val).

    train_idx

        Indices of train data.

    val_idx

        Indices of val data.

    batch_size

        Number of rows to include in each batch.



    Returns

    -------

    dict

        Keys are `train` and `val`, values are respective dataloaders.

    """

    train_df = df.iloc[train_idx]

    val_df = df.iloc[val_idx]



    train_loader = torch.utils.data.DataLoader(

        TweetDataset(train_df),

        batch_size=batch_size,

        shuffle=True,

        num_workers=2,

        drop_last=True,

    )



    val_loader = torch.utils.data.DataLoader(

        TweetDataset(val_df), batch_size=batch_size, shuffle=False, num_workers=2,

    )



    dataloaders_dict = {"train": train_loader, "val": val_loader}



    return dataloaders_dict





def get_test_loader(df: pd.DataFrame, batch_size: int = 32) -> torch.utils.data.DataLoader:

    """

    Get test data loader.



    Parameters

    ----------

    df

        Entire testing dataset.

    batch_size

        Number of rows to include in each batch.



    Returns

    -------

    DataLoader

    """

    loader = torch.utils.data.DataLoader(

        TweetDataset(df), batch_size=batch_size, shuffle=False, num_workers=2

    )

    return loader
class TweetModel(nn.Module):

    """

    Model for Tweet dataset.



    Structure is:

    - Pretrained Roberta base

    - dropout

    - fully connected layer with 2 outputs

    """



    def __init__(self) -> None:

        """Initialise TweetModel."""

        super(TweetModel, self).__init__()



        self.roberta = RobertaModel.from_pretrained(

            "../input/roberta-base", output_hidden_states=True

        )

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(self.roberta.config.hidden_size, 2)

        nn.init.normal_(self.fc.weight, std=0.02)

        nn.init.normal_(self.fc.bias, 0)



    def forward(

        self, input_ids: torch.Tensor, attention_mask: torch.Tensor

    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """

        Compute forward-pass when training model.



        Parameters

        ----------

        input_ids

            Batch of encoded tweets

        attention mask

            Batch of attention masks, telling us which tokens are outside the padding.



        Returns

        -------

        start_logits

            Predictions for where the selected text should start

        end_logits

            Predictions for where the selected text should end

        """

        assert input_ids.shape[1:] == torch.Size([96])

        assert attention_mask.shape[1:] == torch.Size([96])



        hs: Tuple[torch.Tensor, ...]

        _, _, hs = self.roberta(input_ids, attention_mask)



        assert len(hs) == 13

        assert all(i.shape[1:] == torch.Size([96, 768]) for i in hs)



        x = torch.stack([hs[-1], hs[-2], hs[-3], hs[-4]])

        assert x.shape[0] == 4

        assert x.shape[2:] == torch.Size([96, 768])



        x = torch.mean(x, 0)

        assert x.shape[1:] == torch.Size([96, 768])



        x = self.dropout(x)

        x = self.fc(x)

        assert x.shape[1:] == torch.Size([96, 2])



        start_logits, end_logits = x.split(1, dim=-1)

        assert all(i.shape[1:] == torch.Size([96, 1]) for i in [start_logits, end_logits])



        start_logits = start_logits.squeeze(-1)

        end_logits = end_logits.squeeze(-1)

        assert all(i.shape[1:] == torch.Size([96]) for i in [start_logits, end_logits])



        return start_logits, end_logits
def loss_fn(

    start_logits: torch.Tensor,

    end_logits: torch.Tensor,

    start_positions: torch.Tensor,

    end_positions: torch.Tensor,

) -> torch.Tensor:

    """

    Calculate cross-entropy losses for start and end token predictions and sum them.



    Parameters

    ----------

    start_logits

        Logits for prediction of starting token.

    end_logits

        Logits for prediction of ending token.

    start_positions

        Ground truth starting token.

    end_positions

        Ground truth ending token.



    Returns

    -------

    torch.Tensor

        start loss + end loss

    """

    assert start_logits.shape[1:] == torch.Size([96])

    assert end_logits.shape[1:] == torch.Size([96])

    assert start_positions.shape[1:] == torch.Size([])

    assert end_positions.shape[1:] == torch.Size([])



    ce_loss = nn.CrossEntropyLoss()



    start_loss = ce_loss(start_logits, start_positions)

    assert start_loss.shape == torch.Size([])



    end_loss = ce_loss(end_logits, end_positions)

    assert end_loss.shape == torch.Size([])



    total_loss = start_loss + end_loss

    return total_loss
def get_selected_text(text: str, start_idx: int, end_idx: int, offsets: np.array) -> str:

    """

    Extract selected text from tweet.



    Parameters

    ----------

    text

        Lowercase tweet with leading whitespace

    start_idx

        Token where selected text starts

    end_idx

        Token where selected text ends

    offsets

        Positions of text corresponding to each token



    Returns

    -------

    str

        Selected text with leading blank space



    Examples

    --------

    >>> text = " i love python"

    >>> start_idx = 2

    >>> end_idx = 2

    >>> offsets = np.array([[0, 2], [2, 7], [7, 14]] + [[0, 0]] * 93)

    >>> get_selected_text(text, start_idx, end_idx, offsets)

    ' python'

    """

    assert offsets.shape == (96, 2)



    selected_text = ""

    for ix in range(start_idx, end_idx + 1):

        selected_text += text[offsets[ix][0] : offsets[ix][1]]

        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:

            selected_text += " "

    return selected_text





def jaccard(str1: str, str2: str) -> float:

    """

    Compute Jaccard Score of two strings.



    This is given by the intersection over the union.



    Parameters

    ----------

    str1

        First string.

    str2

        Second string.



    Returns

    -------

    float

        Jaccard score.

    """

    a = set(str1.lower().split())

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))





def compute_jaccard_score(

    text: str,

    start_idx: int,

    end_idx: int,

    start_logits: np.array,

    end_logits: np.array,

    offsets: np.array,

) -> float:

    """

    Compute Jaccard score for one prediction.



    Parameters

    ----------

    text

        Lowercase tweet with leading whitespace.

    start_idx

        Ground truth starting token for selected text.

    end_idx

        Ground truth ending token for selected text.

    start_logits

        Predicted logits for start idx.

    end_logits

        Predicted logits for end idx.

    offsets

        Positions of text corresponding to tokens.



    Returns

    -------

    float

        Jaccard score.

    """

    assert start_logits.shape == (96,)

    assert end_logits.shape == (96,)

    assert offsets.shape == (96, 2)



    start_pred = np.argmax(start_logits)

    end_pred = np.argmax(end_logits)



    if start_pred > end_pred:

        pred = text

    else:

        pred = get_selected_text(text, start_pred, end_pred, offsets)



    true = get_selected_text(text, start_idx, end_idx, offsets)



    return jaccard(true, pred)
def loop_through_data_loader(

    model: TweetModel,

    dataloaders_dict: Dict[str, torch.utils.data.DataLoader],

    optimizer: optim.AdamW,

    num_epochs: int,

    epoch: int,

    phase: str,

) -> TweetModel:

    """

    Train on training data or make prediction for validation data.



    Parameters

    ----------

    model

        The NLP model, which may already have been partially trained on a previous

        epoch.

    dataloaders_dict

        Train and val dataloaders

    optimizer

        Adapts learning rate for each weight.

    num_epochs

        Total number of epochs we're training for

    epoch

        Current epoch

    phase

        Train or val



    Returns

    -------

    model

    """

    assert phase in ["train", "val"]



    epoch_loss = 0.0

    epoch_jaccard = 0.0



    for data in dataloaders_dict[phase]:

        ids = data["ids"].cuda()

        masks = data["masks"].cuda()

        tweet = data["tweet"]

        offsets = data["offsets"].numpy()

        start_idx = data["start_idx"].cuda()

        end_idx = data["end_idx"].cuda()



        optimizer.zero_grad()



        with torch.set_grad_enabled(phase == "train"):



            start_logits, end_logits = model(ids, masks)



            loss = loss_fn(start_logits, end_logits, start_idx, end_idx)



            if phase == "train":

                loss.backward()

                optimizer.step()



            epoch_loss += loss.item() * len(ids)



            start_idx = start_idx.cpu().detach().numpy()

            end_idx = end_idx.cpu().detach().numpy()

            start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()

            end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()



            for i in range(len(ids)):

                jaccard_score = compute_jaccard_score(

                    tweet[i],

                    start_idx[i],

                    end_idx[i],

                    start_logits[i],

                    end_logits[i],

                    offsets[i],

                )

                epoch_jaccard += jaccard_score



    epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)

    epoch_jaccard = epoch_jaccard / len(dataloaders_dict[phase].dataset)



    print(

        "Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}".format(

            epoch + 1, num_epochs, phase, epoch_loss, epoch_jaccard

        )

    )



    return model
def train_one_epoch(

    model: TweetModel,

    dataloaders_dict: Dict[str, torch.utils.data.DataLoader],

    optimizer: optim.AdamW,

    num_epochs: int,

    epoch: int,

) -> TweetModel:

    """

    Train on training data for one epoch.



    Parameters

    ----------

    model

        The NLP model, which may already have been partially trained on a previous

        epoch.

    dataloaders_dict

        Train and val dataloaders

    optimizer

        Adapts learning rate for each weight.

    num_epochs

        Total number of epochs we're training for

    epoch

        Current epoch



    Returns

    -------

    model

        Trained for one extra epoch.

    """

    # train

    model.train()

    model = loop_through_data_loader(

        model, dataloaders_dict, optimizer, num_epochs, epoch, "train",

    )



    # evaluate

    model.eval()

    model = loop_through_data_loader(

        model, dataloaders_dict, optimizer, num_epochs, epoch, "val"

    )



    return model
def train_model(

    model: TweetModel,

    dataloaders_dict: Dict[str, torch.utils.data.DataLoader],

    optimizer: optim.AdamW,

    num_epochs: int,

    filename: str,

) -> None:

    """

    Train on training data for given number of epochs, save model weights.



    Parameters

    ----------

    model

        The NLP model, which may already have been partially trained on a previous

        epoch.

    dataloaders_dict

        Train and val dataloaders

    optimizer

        Adapts learning rate for each weight.

    num_epochs

        Total number of epochs we're training for

    filename

        Where to save model weights.

    """

    model.cuda()



    for epoch in range(num_epochs):

        model = train_one_epoch(model, dataloaders_dict, optimizer, num_epochs, epoch)



    torch.save(model.state_dict(), filename)
%%time



if __name__ == "__main__":



    if not RUN_LOCAL:

        train_df = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")

    else:

        train_df = pd.read_csv("../input/tweet-sentiment-extraction/train.csv", nrows=10)

    train_df["text"] = train_df["text"].astype(str)

    train_df["selected_text"] = train_df["selected_text"].astype(str)



    for fold, (train_idx, val_idx) in enumerate(

        SKF.split(train_df, train_df.sentiment), start=1

    ):

        print(f"Fold: {fold}")



        model = TweetModel()

        optimizer = optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.999))

        dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, BATCH_SIZE)



        train_model(

            model, dataloaders_dict, optimizer, NUM_EPOCHS, f"roberta_fold{fold}.pth",

        )
%%time



if __name__ == "__main__":

    if not RUN_LOCAL:

        test_df = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")

    else:

        test_df = pd.read_csv("../input/tweet-sentiment-extraction/test.csv", nrows=1)

    test_df["text"] = test_df["text"].astype(str)

    test_loader = get_test_loader(test_df)

    predictions = []

    models = []

    for fold in range(SKF.n_splits):

        model = TweetModel()

        model.cuda()

        model.load_state_dict(torch.load(f"./roberta_fold{fold+1}.pth"))

        model.eval()

        models.append(model)



    for data in test_loader:

        ids = data["ids"].cuda()

        masks = data["masks"].cuda()

        tweet = data["tweet"]

        offsets = data["offsets"].numpy()



        start_logits = []

        end_logits = []

        for model in models:

            with torch.no_grad():

                output = model(ids, masks)

                start_logits.append(torch.softmax(output[0], dim=1).cpu().detach().numpy())

                end_logits.append(torch.softmax(output[1], dim=1).cpu().detach().numpy())



        start_logits = np.mean(start_logits, axis=0)

        end_logits = np.mean(end_logits, axis=0)

        for i in range(len(ids)):

            start_pred = np.argmax(start_logits[i])

            end_pred = np.argmax(end_logits[i])

            if start_pred > end_pred:

                pred = tweet[i]

            else:

                pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])

            predictions.append(pred)
if __name__ == "__main__":

    if not RUN_LOCAL:

        sub_df = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")

    else:

        sub_df = pd.read_csv(

            "../input/tweet-sentiment-extraction/sample_submission.csv", nrows=1

        )

    sub_df["selected_text"] = predictions

    sub_df["selected_text"] = sub_df["selected_text"].apply(

        lambda x: x.replace("!!!!", "!") if len(x.split()) == 1 else x

    )

    sub_df["selected_text"] = sub_df["selected_text"].apply(

        lambda x: x.replace("..", ".") if len(x.split()) == 1 else x

    )

    sub_df["selected_text"] = sub_df["selected_text"].apply(

        lambda x: x.replace("...", ".") if len(x.split()) == 1 else x

    )

    sub_df.to_csv("submission.csv", index=False)

    sub_df.head()