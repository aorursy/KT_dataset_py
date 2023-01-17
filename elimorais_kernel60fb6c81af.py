# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
dados_df = pd.read_excel('/kaggle/input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')
dados_df.head()
dados_df['AGE_PERCENTIL']
dados_df['AGE_PERCENTIL'].value_counts()
dados_df.query("ICU == 1")
dados_df.groupby("PATIENT_VISIT_IDENTIFIER", as_index = False).agg({"ICU":(list), "WINDOW": list}).iloc[[13,14,15,41,0,2]]
aux = abs(dados_df.groupby("PATIENT_VISIT_IDENTIFIER")["ICU"].sum()-5)
aux = aux.value_counts().reset_index()
aux.sort_values(by = "index", inplace = True)
aux.reset_index(drop = True, inplace = True)
import matplotlib.pyplot as plt

tot_icu_inpatients = aux.ICU[0:5].sum()
y=aux.ICU[0:5].cumsum()/tot_icu_inpatients
plt.plot(y, marker = ".")



import matplotlib.pyplot as plt

tot_icu_inpatients = aux.ICU[0:5].sum()
y=aux.ICU[0:5].cumsum()/tot_icu_inpatients
plt.plot(y, marker = ".")

# em que momento o paciente vai pra UTI?
plt.ylabel('porcentagem')
plt.xlabel('em horas')
plt.yticks(round(y,2) )
plt.xticks([0,1,2,3,4], ["0-2", "2-4", "4-6", "6-12", "Above-12"])
plt.show
