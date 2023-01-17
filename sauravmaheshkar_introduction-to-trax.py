# First, we install the library

!pip install trax
import trax

from trax import layers as tl
mode = "train"



# Arbitrary Values

vocab_size = 128

model_dimension = 256

n_layers = 2 
GRU = tl.Serial(

    tl.ShiftRight(mode = mode),

    tl.Embedding(vocab_size = vocab_size, d_feature = model_dimension),

    [tl.GRU(n_units=model_dimension) for _ in range(n_layers)],

    tl.Dense(n_units = vocab_size),

    tl.LogSoftmax()

)