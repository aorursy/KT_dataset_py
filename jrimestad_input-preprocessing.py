# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# names of the given files

input = ["biology.csv", "cooking.csv", "crypto.csv", "diy.csv", "robotics.csv", "travel.csv"]



# Import bag of words

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from bs4 import BeautifulSoup  

from nltk.corpus import stopwords



for train_file in input:

    # each file has id, title, content, tags

    data = pd.read_csv("../input/" + train_file)

    print("Parsed file")

    vectorizer = CountVectorizer(min_df=1, preprocessor=None, stop_words=stopwords.words("english"))

    title_data = data.title.values

    counts = vectorizer.fit_transform(title_data)

    print("CountVecter created")

    transformer = TfidfTransformer(smooth_idf=False)

    tfidfCounts = transformer.fit_transform(counts)

    print("TFIDF count computed.")

    print(tfidCounts)

    break
