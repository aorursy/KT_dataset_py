

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

!pip install --upgrade bamboolib>=1.2.0
import bamboolib as bam

bam.enable()
netflix = pd.read_csv("/kaggle/input/netflix-shows/netflix_titles_nov_2019.csv")
bam.show(netflix)