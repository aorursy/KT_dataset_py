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
import codecs

import json
with codecs.open('../input/most_popular_quotes.json',"r","utf-8") as f:

    data = f.read()

data = json.loads(data)
data = pd.DataFrame(data)
data.head()
data["likes"] = data.likes.apply(lambda _: _.split(" ")[0]).astype(int)
pd.DataFrame({"Author":data.author.value_counts().index,"quote_likes":data.author.value_counts().values})
data.sort_values("likes",ascending=False).ix[0:30,:]
authors_of_quotes_with_most_votes = list(data.sort_values("likes",ascending=False).ix[0:30,:].author)