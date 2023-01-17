shakespeare_file_path = "/kaggle/input/shakespeare_complete_works.txt"
!pip install gpt-2-finetuning==0.10
# Available sizes: 124M, 355M, 774M

!download_gpt2_model 355M
MODEL = '355M'
import os

import tensorflow as tf

import numpy as np



from gpt_2_finetuning.interactive_conditional_samples import interact_model

from gpt_2_finetuning.train import train
train(dataset_path=shakespeare_file_path,

      model_name=MODEL,

      n_steps=10000,

      save_every=5000,

      sample_every=1000,

      mem_saving_gradients=True,

      print_loss_every=1000,

      max_checkpoints_to_keep=2)
!rm -rf models
!ls
!ls checkpoint/run1
!nvidia-smi
## Interact example

# interact_model(model_name=MODEL,

#                length=100,

#                top_k=40)
## Encode example

# from gpt_2_finetuning.load_dataset import load_dataset

# from gpt_2_finetuning.encoder import get_encoder



# enc = get_encoder(model_name)

# chunks = load_dataset(enc, shakespeare_file_path, combine=50000, encoding='utf-8')

# enc.encode("PyCon is awesome")

# enc.decode([20519])