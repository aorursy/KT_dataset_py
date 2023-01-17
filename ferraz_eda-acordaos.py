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



# Any results you write to the current directory are saved as output.
import sqlite3

import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

import seaborn as sns

%matplotlib inline
rc={'savefig.dpi': 75, 'figure.autolayout': False, 'figure.figsize': [12, 8], 'axes.labelsize': 18,\

   'axes.titlesize': 18, 'font.size': 18, 'lines.linewidth': 2.0, 'lines.markersize': 8, 'legend.fontsize': 16,\

   'xtick.labelsize': 16, 'ytick.labelsize': 16}



sns.set(style='white',rc=rc)



default_color = '#56B4E9'

colormap = plt.cm.cool
def initiate_db(strcnx: Path) -> sqlite3.Cursor:

    """

    Conecta no banco sqlite



    Atributos:

        strcnx: string de conexão.

    """

    strcnx_is_valid = Path(strcnx)

    if not strcnx_is_valid.is_file():

        raise ("O arquivo sqlite3 não existe.")

    conn = sqlite3.connect(str(strcnx))

    cur = conn.cursor()

    return conn, cur
db_path = Path("/kaggle/input/acordaos-tcu/tcu-acordaos.db")

conn, cur = initiate_db(db_path)
query = "SELECT COUNT(DISTINCT id) as quantitativo_acordao, ano_acordao as ano from acordaos group by ano_acordao"

count_acordaos = pd.read_sql_query(query, conn)
count_acordaos
count_acordaos['ano'] = count_acordaos["ano"].astype('category')
count_acordaos.dtypes
count_acordaos.plot.bar(x='ano')
query_relator = "SELECT COUNT(DISTINCT id) as quantitativo_acordao, relator from acordaos group by relator"

count_relator = pd.read_sql_query(query_relator, conn)
count_relator.sort_values('quantitativo_acordao',inplace=True, ascending=False)
plt.figure(figsize=(12,15)) 

ax = sns.barplot(data=count_relator, x= "quantitativo_acordao", y = count_relator.relator,  palette="GnBu_d") 

ax.set(ylabel = 'Relator');
query_dados_2016 = "SELECT * from acordaos where ano_acordao = 2016"

d2016 = pd.read_sql_query(query_dados_2016, conn)
d2016.head()