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
clima = pd.read_csv("../input/datasets-tcc-separados/clima.csv")

telhado_verde_pivot = pd.read_csv("../input/datasets-tcc-separados/telhado_verde_pivot.csv")
telhado_verde_pivot['Data'] = pd.to_datetime(telhado_verde_pivot['data'])

telhado_verde_pivot['Data'] = telhado_verde_pivot['Data'].dt.strftime('%Y-%m-%d')
clima['Data'] = pd.to_datetime(clima['data'])

clima['Data'] = clima['Data'].dt.strftime('%Y-%m-%d')
clima_dia = clima[clima['Dia'] == 1]

telhado_dia = telhado_verde_pivot[telhado_verde_pivot['Dia'] == 1]
telhado_verde_pivot
datas_telhado12 = pd.DataFrame(telhado_verde_pivot.groupby(['Delta_1_2'])['data'].count())

#datas_telhado.sort_values(['data'],ascending=False)

datas_telhado12[datas_telhado12['data'] > 10].sort_values(['data'],ascending=False)
datas_telhado34 = pd.DataFrame(telhado_verde_pivot.groupby(['Delta_3_4'])['data'].count())

#datas_telhado.sort_values(['data'],ascending=False)

datas_telhado34[datas_telhado34['data'] > 10].sort_values(['data'],ascending=False)
datas_telhado56 = pd.DataFrame(telhado_verde_pivot.groupby(['Delta_5_6'])['data'].count())

#datas_telhado.sort_values(['data'],ascending=False)

datas_telhado56[datas_telhado56['data'] > 10].sort_values(['data'],ascending=False)
telhado_verde_pivot[telhado_verde_pivot['Delta_1_2'] == -7.0]