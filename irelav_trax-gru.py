#!pip install trax
import trax
import trax.fastmath.numpy as np
import numpy
import random as rnd
from trax import fastmath
from trax import layers as tl
from trax import shapes

# set random seed

rnd.seed(32)
x = np.array([[1, 1, 2, 3, 4],[1, 2, 3 , 4, 5]])
print(f"Some tokenized data:\n {x} \n of shape {x.shape}. This is e.g. a batch of 2 sentences with 5 words each after tokenization.")
shifter = tl.ShiftRight(mode="train")
shifted = shifter(x)
print(f"Right-shifted data:\n {shifted} \n of shape {shifted.shape}.")
emb_dim = 4
embed = tl.Embedding(vocab_size=10, d_feature=emb_dim)
_, _ = embed.init(None)
embedded = embed(y)
print(f"After embedding each word with a 4-dimentsional embedding we get:\n {embedded} \n of shape {embedded.shape} - 2 baches of 5 words, each represented as a 4-dimensional embedding.")
grucell = tl.GRUCell(n_units=emb_dim)
embedded_2 = (embedded, embedded)
_, _ = grucell.init(shapes.signature(embedded_2))
grus = grucell(embedded_2)
print(f"Output of the gru cell is \n\n {grus} \n\nwhich is of type {type(grus)} of lenght {len(grus)}.")
print(f"Each element of the tuple is of shape {grus[0].shape}.")
grucell_n_2 = tl.GRUCell(emb_dim-2)
x_and_h = (z, np.ones_like(z[:,:,:2])) # the first element is the input x_t and the second is h_{t-1} which we have now reduced to a dimension of 2 instead of 4 (the embedding dimension)
_, _ = grucell_n_2.init(shapes.signature(x_and_h))
grus_n_2 = grucell_n_2(x_and_h)
grus_n_2
print(f"Output of the gru cell is \n\n {grus_n_2} \n\nwhich is of type {type(grus_n_2)} of lenght {len(grus_n_2)}.")
print(f"Each element of the tuple is of shape {grus_n_2[0].shape} - different from the previous one which had a shape of {grus[0].shape}.")