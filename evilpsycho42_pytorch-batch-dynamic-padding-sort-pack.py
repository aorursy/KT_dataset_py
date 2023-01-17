import torch

from torch.utils import data

from torch.nn.utils.rnn import pack_padded_sequence

from keras.preprocessing import sequence

import numpy as np # linear algebra
# truncated sentence by lens percentile of batch

PERCENTILE = 80

# truncated sentence by a constant

MAX_LEN = 4

BATCH_SIZE = 4
class MyDataset(data.Dataset):

    

    def __init__(self, text, lens, y=None):

        self.text = text

        self.y = y

        self.lens = lens

    

    def __len__(self):

        return len(self.lens)

    

    def __getitem__(self, index):

        if self.y is None:

            return self.text[index], self.lens[index]

        else:

            return self.text[index], self.lens[index], self.y[index]

    



def collate_fn(batch):

    """

    batch = [dataset[i] for i in N]

    """

    size = len(batch[0])

    if size == 3:

        texts, lens, y = zip(*batch)

    else:

        texts, lens = zip(*batch)

    lens = np.array(lens)

    sort_idx = np.argsort(-1 * lens)

    reverse_idx = np.argsort(sort_idx)

    max_len = min(int(np.percentile(lens, PERCENTILE)), MAX_LEN)

    

    lens = np.clip(lens, 0, max_len)[sort_idx]

    texts = torch.tensor(sequence.pad_sequences(texts, maxlen=max_len)[sort_idx], dtype=torch.long)

    if size == 3:

        return texts, lens, reverse_idx, torch.tensor(y, dtype=torch.float32)

    else:

        return texts, lens, reverse_idx





def build_data_loader(texts, lens, y=None, batch_size=BATCH_SIZE):

    dset = MyDataset(texts, lens, y)

    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return dloader
seqs = [[1,2,3,3,4,5,6,7], [1,2,3], [2,4,1,2,3], [1,2,4,1]]

lens = [len(i) for i in seqs]



data_loader = build_data_loader(seqs, lens)



for batch in data_loader:

    seq_batch, lens_batch, reverse_idx_batch = batch

    break
print(f'original seqs:')

print(seqs)

print(f'batch seqs, already sort by lens, and padding dynamic in batch:')

print(seq_batch.numpy().tolist())

print(f'reverse batch seqs:')

print(seq_batch[reverse_idx_batch].numpy().tolist())
pack_padded_sequence(seq_batch, lens_batch, batch_first=True)