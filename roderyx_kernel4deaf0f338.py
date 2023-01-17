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

        if filename == "en_US.news.txt":

            with open(os.path.join(dirname,filename)) as file:

                news = file.read()

        elif filename == "en_US.twitter.txt":

            with open(os.path.join(dirname,filename)) as file:

                tweets = file.read()

        elif filename == "en_US.blogs.txt":

            with open(os.path.join(dirname,filename)) as file:

                blogs = file.read()



# Any results you write to the current directory are saved as output.
# Transformacion de mayusculas a minusculas

tweets = tweets.lower()

blogs = blogs.lower()

news = news.lower()

print(news.split(' ')[0])
# Eliminacion de simbolos observados

tweets = tweets.strip("<")

blogs = blogs.strip("<")

news = news.strip("<")

tweets = tweets.strip(">")

blogs = blogs.strip(">")

news = news.strip(">")

tweets = tweets.strip("-")

blogs = blogs.strip("-")

news = news.strip("-")

tweets = tweets.strip("#")

blogs = blogs.strip("#")

news = news.strip("#")

tweets = tweets.strip("*")

blogs = blogs.strip("*")

news = news.strip("*")

# Remover simboloes tales como emoticonos

tweets = tweets.encode('ascii', 'ignore')

blogs = blogs.encode('ascii', 'ignore')

news = news.encode('ascii', 'ignore')