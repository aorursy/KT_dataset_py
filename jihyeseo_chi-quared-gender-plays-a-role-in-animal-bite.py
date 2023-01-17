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
df = pd.read_csv('../input/Health_AnimalBites.csv')
df.dtypes
df.head()
df.head_sent_date = pd.to_datetime(df.head_sent_date, infer_datetime_format = True, errors = 'coerce')
df.release_date = pd.to_datetime(df.release_date, infer_datetime_format = True, errors = 'coerce')
df.quarantine_date = pd.to_datetime(df.quarantine_date, infer_datetime_format = True, errors = 'coerce')
df.vaccination_date = pd.to_datetime(df.vaccination_date, infer_datetime_format = True, errors = 'coerce')
df.bite_date = pd.to_datetime(df.bite_date, infer_datetime_format = True, errors = 'coerce')

df.describe(include = 'O').transpose()
df.describe(exclude = 'O').transpose()
df.SpeciesIDDesc.value_counts().plot.bar()
df.GenderIDDesc.value_counts().plot.bar()
df.columns
df = df[df.SpeciesIDDesc.isin(['DOG','CAT']) & df.GenderIDDesc.isin(['MALE','FEMALE'])]
# with help from https://www.kaggle.com/omarayman/chi-square-test-in-python

cont = pd.crosstab(df["SpeciesIDDesc"],df["GenderIDDesc"])
    
cont
import scipy
scipy.stats.chi2_contingency(cont)