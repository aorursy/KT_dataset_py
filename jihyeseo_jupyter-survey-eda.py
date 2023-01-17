# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import math
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


from wordcloud import WordCloud, STOPWORDS

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input"]).decode("utf8").strip()
# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)
df = pd.read_csv('../input/Survey.csv')
df.shape
df.dtypes
df.columns
df['How do you run the Jupyter Notebook?'].value_counts().plot.bar()
df['Roughly how long have you been using Jupyter Notebook?'].value_counts().plot.bar()
#df['What is your primary role when using Jupyter Notebook (e.g., student, astrophysicist, financial modeler, business manager, etc.)?'].value_counts().plot.bar()
df['How many people typically see and/or interact with the results of your work in Jupyter Notebook? (Consider people who view your notebooks on nbviewer, colleagues who rerun your notebooks, developers who star your notebook repos on GitHub, audiences who see your notebooks as slideshows, etc.)'].value_counts().plot.bar()
df['How do you run the Jupyter Notebook?'].value_counts().plot.bar()


df['How often do you use Jupyter Notebook?'].value_counts().plot.bar()
