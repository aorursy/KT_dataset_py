



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



!pip install gpt-2-simple
!pip install tensorflow-gpu==1.14.0
import gpt_2_simple as gpt2
model_name = "355M"

if not os.path.isdir(os.path.join("models", model_name)):

    print(f"Downloading {model_name} model...")

    gpt2.download_gpt2(model_name=model_name)

file_name = "/kaggle/input/songs-lyrics-in-text-file/songs.txt"



sess = gpt2.start_tf_sess()

gpt2.finetune(sess,

              file_name,

              model_name=model_name,

              steps=450)   # steps is max number of training steps

gpt2.generate(sess)