import pandas as pd

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from IPython.display import Image

import os
Image("/kaggle/input/week9dataset/Introduction_to_Reinforcement_Learning1.jpeg")
Image("/kaggle/input/week9dataset/Introduction to Reinforcement Learning2.png")
Image("/kaggle/input/week9dataset/Introduction to Reinforcement Learning3.png")
Image("/kaggle/input/week9dataset/Introduction to Reinforcement Learning4.png")
Image("/kaggle/input/week9dataset/Introduction to Reinforcement Learning5.png")