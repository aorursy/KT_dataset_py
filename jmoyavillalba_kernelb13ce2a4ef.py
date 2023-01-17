# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
elecciones = pd.read_csv('../input/Datos_Eleciones_Generales_Provincias.csv')

municipios = pd.read_csv('../input/19codmun_en - 19codmun_en (1).csv')

provincias = pd.read_csv('../input/19_cod_prov - 19_cod_prov (1).csv')

comunidades = pd.read_csv('../input/19_cod_ccaa - 19_cod_ccaa (2).csv')
provincias['CPRO'] = provincias['CODES']

comunidades['CODAUTO'] = comunidades['CODES']

muni_auto = pd.merge(municipios, comunidades, on='CODAUTO', how='left')

prov_muni_auto = pd.merge(muni_auto, provincias, on='CPRO', how='left')

prov_muni_auto['IDPROV'] = prov_muni_auto['CPRO']

full_elec = pd.merge(prov_muni_auto, elecciones, on='IDPROV', how='left')

full_elec.head()
full_elec['MEDIA_CENSO'] = (full_elec['Censo1996'] + full_elec['Censo2000'] + full_elec['Censo2004'] + full_elec['Censo2008'] + full_elec['Censo2011'] + full_elec['Censo2016'])/6

# media = full_elec.groupby(['CODAUTO'])

x = full_elec[['CODAUTO', 'MEDIA_CENSO']]

x = x.drop_duplicates()

x.groupby('CODAUTO')['MEDIA_CENSO'].mean()