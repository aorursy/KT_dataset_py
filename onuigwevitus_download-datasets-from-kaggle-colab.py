# Install kaggle to colab

!pip install kaggle
from google.colab import files

files.upload()
!mkdir -p ~/.kaggle

!cp kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json
# ! kaggle competitions download -c 'name-of-competition'

# Example

! kaggle competitions download -c 'digit-recognizer'
!unzip /content/train.csv.zip

!unzip /content/test.csv.zip
# LOAD THE DATA and start your competitions or learning process

import pandas as pd



train = pd.read_csv("/content/train.csv")

test = pd.read_csv("/content/test.csv")