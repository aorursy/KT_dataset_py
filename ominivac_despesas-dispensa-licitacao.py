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
df = pd.read_csv('/kaggle/input/despesas_empenho_01_31.csv', encoding = 'utf-8', sep=';')

df.head()

df.dtypes

df.columns
new_columns =  ['id_emp', 'cod_emp', 'cod_empenho_res',
       'dt_emissão', 'cod_tp_doc', 'tp_doc',
       'tp_empenho', 'esp_empenho', 'cod_org_sup',
       'org_sup', 'cod_orgao', 'orgao', 'cod_uni_gest',
       'uni_gest', 'cog_gestao', 'gestao', 'cod_fav',
       'fav', 'obs', 'cod_esf_orc',
       'esf_orc', 'cod_tp_cred', 'tp_cred',
       'cod_gr_fonte_rec', 'gr_fonte_rec',
       'cod_fonte_rec', 'fonte_rec', 'cod_un_orc',
       'un_orc', 'cod_funcao', 'funcao', 'cod_sub_funcao',
       'sub_funcao', 'cod_programa', 'programa', 'cod_acao', 'acao',
       'ling_cidada', 'cod_subtitulo_loc',
       'subtitulo_loc', 'cod_plano_orc',
       'plano_orc', 'autor_emenda', 'cod_cat_despesa',
       'cat_despesa', 'cod_gr_desp', 'gr_desp',
       'cod_mod_aplic', 'mod_aplic',
       'cod_elem_desp', 'elem_desp', 'processo',
       'mod_lic', 'inciso', 'amparo',
       'ref_disp_inex', 'cod_conv',
       'contrato_repasse',
       'vlr_org_emp', 'vlr_org_emp_real',
       'vlr_uti_conver']

df.columns =  new_columns
df.head()
df["vlr_org_emp"] = df.vlr_org_emp.astype(float)
null_columns = df.columns[df.isnull().any()]
df[null_columns].isnull().sum()
df.loc['vlr_org_emp',:]= df.sum(axis=0)
df.sum(axis = 1, skipna = True) 
print('total',total)