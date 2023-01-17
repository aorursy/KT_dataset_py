text = '''@RELATION AND

@ATTRIBUTE X1 {0, _}

@ATTRIBUTE X2 {0, _}

@ATTRIBUTE Y {0, _}

@DATA

_, _, _

_, _, _

_, _, _

_, _, _

'''



with open('AND.arff','w') as file:

  file.write(text)
from scipy.io import arff

import pandas as pd



data = arff.loadarff('AND.arff')

df = pd.DataFrame(data[0])



df.head()
from sklearn import datasets # see https://scikit-learn.org/stable/datasets/index.html



iris = datasets.load_iris()

iris # to print out the content
print(iris.DESCR)
import pandas as pd

import numpy as np # 



iris_pd = pd.DataFrame(data = np.c_[iris['data'], iris['target']], columns = iris['feature_names']+['target'])

iris_pd
from sklearn.datasets import fetch_openml

weather = fetch_openml(data_id=41521)

weather
import textwrap

print(textwrap.fill(weather.DESCR,100))
import pandas as pd

import numpy as np



weather_pd = pd.DataFrame(data=np.c_[weather.data,weather.target],columns=weather.feature_names+['target'])

weather_pd
import urllib.request

import io

import pandas as pd



url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

ftpstream = urllib.request.urlopen(url)

iris = pd.read_csv(io.StringIO(ftpstream.read().decode('utf-8')))

iris