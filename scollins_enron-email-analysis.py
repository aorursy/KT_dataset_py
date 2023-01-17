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
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
#import TextBlob
%matplotlib inline
df = pd.read_csv('../input/emails.csv')
df.head()
# See what the columns are
df.columns
# Create arrays to store different parts of the filenames
users = []
mailtype = []
filenumber = []

# Separate the file fields
def parse_filenames(files):
    for x in range(0, len(files)):
        a, b, c = files[x].split('/')
        users.append(a)
        mailtype.append(b)
        filenumber.append(c)
# Run the parse filenames function
#parse_filenames(df.file)
# Look at the text
df.message[0]
# More to come later.
