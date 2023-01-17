# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go

import plotly.offline as py

import plotly.express as px





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/serratus-ultrahigh-throughput-viral-discovery/doc/zoonotic_candidates.tsv', sep='\t', error_bad_lines=False)

df.head()
df.isnull().sum()
def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
fig = px.parallel_categories(df, color="30684", color_continuous_scale=px.colors.sequential.Cividis)

fig.show()
plot_count("Porcine hemagglutinating encephalomyelitis virus strain JL/2008, complete genome.", "Complete Genome", df,4)
plt.figure(figsize=(20,4))

plt.subplot(131)

sns.countplot(x= 'Porcine hemagglutinating encephalomyelitis virus strain JL/2008, complete genome.', data = df, palette="gist_stern",edgecolor="black")

plt.xticks(rotation=45)

plt.subplot(132)

sns.countplot(x= 'Viruses;Riboviria;Nidovirales;Cornidovirineae;Coronaviridae;Orthocoronavirinae;Betacoronavirus;Embecovirus.', data = df, palette="gnuplot",edgecolor="black")

plt.xticks(rotation=45)

plt.show()
sns.countplot(x="30684",data=df,palette="GnBu_d",edgecolor="black")

plt.title('30684', weight='bold')

plt.xticks(rotation=45)

plt.yticks(rotation=45)

# changing the font size

sns.set(font_scale=1)