!pip install -U -q sentencepiece ninja

!pip install -U -q git+https://github.com/fastai/fastai
import gc

from functools import partial

from pathlib import Path



from fastai.text import *

from fastai.callbacks import *

import numpy as np

import pandas as pd



home = Path(".")

dest = Path("data/en-100")
!mkdir data

!wget -nc 'https://www.dropbox.com/sh/srfwvur6orq0cre/AAAQc36bcD17C1KM1mneXN7fa/data/wiki?dl=1' -O 'data/preprocessed_wiki_8langs.zip'

!unzip 'data/preprocessed_wiki_8langs.zip' -d 'data'
!tar -xvf data/en-100.tar.gz -C data
(dest/"valid").mkdir(exist_ok=True)

(dest/"train").mkdir(exist_ok=True)

(dest/"test").mkdir(exist_ok=True)

!mv data/en-100/en.wiki.valid.tokens data/en-100/valid/

!mv data/en-100/en.wiki.train.tokens data/en-100/train/

!mv data/en-100/en.wiki.test.tokens data/en-100/test/
from types import FunctionType



def copy_func(f):

    "Copy a non-builtin function (NB `copy.copy` does not work for this)"

    if not isinstance(f,FunctionType): return copy(f)

    fn = FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__)

    fn.__dict__.update(f.__dict__)

    return fn



def patch_to(cls, as_prop=False):

    "Decorator: add `f` to `cls`"

    if not isinstance(cls, (tuple,list)): cls=(cls,)

    def _inner(f):

        for c_ in cls:

            nf = copy_func(f)

            # `functools.update_wrapper` when passing patched function to `Pipeline`, so we do it manually

            for o in functools.WRAPPER_ASSIGNMENTS: setattr(nf, o, getattr(f,o))

            nf.__qualname__ = f"{c_.__name__}.{f.__name__}"

            setattr(c_, f.__name__, property(nf) if as_prop else nf)

        return f

    return _inner



def patch(f):

    "Decorator: add `f` to the first parameter's class (based on f's type annotations)"

    cls = next(iter(f.__annotations__.values()))

    return patch_to(cls)(f)



@patch

def read(self:Path, size=-1, encoding='utf8'):

    "Read the content of `fname`"

    with self.open(encoding=encoding) as f: return f.read(size)
def split_file_into_chunks(path, chunk_size=20480//2, scale=0.25):

    txt = Path(path).read()

#     if len(txt) > 100_000_000:

#         txt = txt[:int(len(txt)*scale)]

    display(f"{len(txt)} tokens")

    prev_chunk = 0

    i = 0

    while True:

        with open(Path(path).parent / f'{i}.txt', "w") as text_file:

            next_chunk = txt[prev_chunk: prev_chunk + chunk_size].rindex("\n")

            if not next_chunk:

                break

            text_file.write(txt[prev_chunk: prev_chunk+next_chunk])

            prev_chunk += next_chunk

            i += 1

    display(f"{i} files")

            

# def split_file_into_chunks(path, num_chunks=48):

#     txt = Path(path).read().split('\n')

#     chunk_len = len(txt) // num_chunks

    

#     for i in range(num_chunks):

#         with open(Path(path).parent / f'{i}.txt', "w") as text_file:

#             text_file.write('\n'.join(txt[i*chunk_len:(i+1)*chunk_len]))
split_file_into_chunks('data/en-100/train/en.wiki.train.tokens')

split_file_into_chunks('data/en-100/valid/en.wiki.valid.tokens')

split_file_into_chunks('data/en-100/test/en.wiki.test.tokens')
# !find data/en-100/ -maxdepth 2 -type f -name "*.txt" -delete 
def get_texts(path):

    rows_list = []   

    for idx, label in enumerate(["train", "valid", "test"]):

        print(f'working on {path}/{label}')

        for fname in (path/f'{label}').glob('*.txt'):

            dict1 = {}

            text = fname.open('r').read()

            dict1.update({

                'text':text,

                'label':idx

            }) 

            rows_list.append(dict1)

        print(len(rows_list))

    df = pd.DataFrame(rows_list)

    return df
df = get_texts(dest)
df = df.drop(df[df.text.str.len() < 100].index).reset_index()
!rm "data/"*
bs=256

max_vocab = 15_000
data_en_wiki = (TextList.from_df(df, dest, ["text"], processor=SPProcessor(mark_fields=False, max_vocab_sz=max_vocab))

                .split_by_rand_pct(0.1, seed=42)

                .label_for_lm()

                .databunch(bs=bs, num_workers=4))



data_en_wiki.save(f"data_en_wiki_{max_vocab}")
len(data_en_wiki.vocab.stoi)
assert len(data_en_wiki.vocab.stoi) == max_vocab
data_en_wiki = (TextList.from_df(df, dest, ["text"], processor=SPProcessor(mark_fields=False, max_vocab_sz=max_vocab))

                .split_by_rand_pct(0.1, seed=42)

                .label_for_lm()

                .databunch(bs=bs, num_workers=4, backwards=True))



data_en_wiki.save(f"data_en_wiki_{max_vocab}_bwd")
!find data/en-100/ -maxdepth 3 -type f -name "*.txt" -delete 
!mv data/en-100/data_en_wiki_15000_bwd data/

!mv data/en-100/data_en_wiki_15000 data/

!mv data/en-100/tmp/ data/