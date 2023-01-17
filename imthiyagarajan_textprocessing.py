# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# Any results you write to the current directory are saved as output.
from shutil import copyfile



copyfile(src = "../input/cleantext/cleantext.py", dst = "../working/cleantext.py")



# import all our functions

from cleantext import *



#!pylint cleantext


import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer



training = [

    " I am master of all",

    "I am a absolute learner"

]



generalization = [

    "I am absolute learner learner"

]



vectorization = CountVectorizer(

    stop_words = "english",

    preprocessor = process.master_clean_text)



vectorization.fit(training)



build_vocab = {

     value:key 

     for key , value in vectorization.vocabulary_.items()

}



vocab = [build_vocab[i] for i in range(len(build_vocab))]



pd.DataFrame(

data = vectorization.transform(generalization).toarray(),

    index=["generalization"],

    columns=vocab

)