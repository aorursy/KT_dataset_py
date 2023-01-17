# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.tree import DecisionTreeClassifier, export_graphviz

from graphviz import Source



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import Normalizer

from sklearn.model_selection import train_test_split





from scipy.stats import skew



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



plt.style.use('ggplot')
df = pd.read_csv("/kaggle/input/ai4all-project/data/viral_calls/sample_overviews.csv")

df.head()
df.isna().sum()
categorical_cols = [col for col in df.columns if df[col].dtype == 'object']

label_encoder = LabelEncoder()

labelled_df = df.copy()

for col in categorical_cols:

    labelled_df[col] = label_encoder.fit_transform(df[col])

labelled_df.head()
import seaborn as sbn



correlation=labelled_df.corr()

plt.figure(figsize=(15,15))

sbn.heatmap(correlation,annot=True,cmap=plt.cm.Greens)
cols_to_drop=['sample_name', 'uploader', 'upload_date', 'overall_job_status', 'host_genome', 'sample_type', 'nucleotide_type', 'collection_date', 'water_control', 'collection_location', 'notes']

df=df.drop(cols_to_drop,axis=1)

df.columns
x = df.drop(['reads_after_trimmomatic', 'reads_after_star'], axis=1)

x.fillna(999999, inplace=True)

y = df['reads_after_star']
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(x, y)
dt_feat = pd.DataFrame(dt.feature_importances_, index=x.columns, columns=['feat_importance'])

dt_feat.sort_values('feat_importance').tail(8).plot.barh()

plt.show()
from IPython.display import SVG

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'



graph = Source(export_graphviz(dt, out_file=None, feature_names=x.columns, filled = True))

display(SVG(graph.pipe(format='svg')))