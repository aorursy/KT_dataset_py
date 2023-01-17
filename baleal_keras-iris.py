# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.models import Sequential

from keras.layers import Dense

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
Tekst = "Nitrous oxide, commonly known as laughing gas or nitrous,[1] is a chemical compound, an oxide of nitrogen with the formula N2O. At room temperature, it is a colorless non-flammable gas, with a slight metallic scent and taste. At elevated temperatures, nitrous oxide is a powerful oxidizer similar to molecular oxygen.Nitrous oxide has significant medical uses, especially in surgery and dentistry, for its anaesthetic and pain reducing effects. Its name laughing gas, coined by Humphry Davy, is due to the euphoric effects upon inhaling it, a property that has led to its recreational use as a dissociative anaesthetic. It is on the World Health Organization's List of Essential Medicines, the most effective and safe medicines needed in a health system.[2] It also is used as an oxidizer in rocket propellants, and in motor racing to increase the power output of engines."

print(Tekst)
import nltk

from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer



def Clean_this(text):

    """ Set text to lower case, removes punctuation, removes stop words"""

    stop_words=set(stopwords.words('english'))

    tokenizer = RegexpTokenizer(r'\w+')

    

    text = text.lower()

    text= tokenizer.tokenize(text)

    filtered_sentence = [w for w in text if not w in stop_words]

    cleaned_string = " ".join(filtered_sentence)

    return(cleaned_string)



Clean_this(Tekst)
## EXTRACT MISSING WORD