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
print(check_output(["ls", "../input/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input/"]).decode("utf8").strip()
# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)
df_d = pd.read_csv('../input/train_date.csv')
dg_d = pd.read_csv('../input/test_date.csv')
df_c = pd.read_csv('../input/train_categorical.csv')
dg_c = pd.read_csv('../input/test_categorical.csv')
df_n = pd.read_csv('../input/train_numeric.csv')
dg_n = pd.read_csv('../input/test_numeric.csv')

dh = pd.read_csv('../input/sample_submission.csv')
