!pip install gpt-2-finetuning
# Available sizes: 124M, 355M, 774M

!download_gpt2_model 774M
MODEL = "774M"
from gpt_2_finetuning.encoder import get_encoder
## The encoder is basically a dictionary mapping of tokens to numbers

enc = get_encoder(MODEL)
enc.encode("PyCon is awesome")
enc.decode([20519])
enc.decode([3103])
enc.decode([318])
enc.decode([7427])
import tensorflow as tf



from gpt_2_finetuning.generate_unconditional_samples import sample_model
sample_model(model_name=MODEL,

             top_k=40,

             nsamples=1,

             batch_size=1)
# cleanup because kaggle notebook seems to be lagging from displaying the encoder json
!rm -rf models