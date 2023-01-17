# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install whatlies
from whatlies import Embedding, EmbeddingSet

from whatlies.language import FasttextLanguage, SpacyLanguage 
ft_lang = FasttextLanguage("/kaggle/input/fasttext-common-crawl-bin-model/cc.en.300.bin")
ft_lang.score_similar("heeello")
!pip install pyspellchecker
from spellchecker import SpellChecker



spell = SpellChecker()



# find those words that may be misspelled

misspelled = spell.unknown(['helloo', 'heeello', 'heeloo', 'hellol'])



for word in misspelled:

    print("misspell Word:",word)

    # Get the one `most likely` answer

    print("Correct Word:",spell.correction(word))



    # Get a list of `likely` options

    print("Other candidates:",spell.candidates(word))