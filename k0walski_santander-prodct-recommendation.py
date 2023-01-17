import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
bank_data = pd.read_csv(

    '../input/train_ver2.csv', 

    sep=',', 

    nrows=1000000, 

    converters={'age': lambda age: None if age.endswith('NA') else int(age)})

bank_data.head()
#bank_data["ind_empleado"].unique()
#Firugre out appropriate columns to filter against NaN later, and do a guess for now

bank_data = bank_data.dropna(axis=0, how='all', subset=['ind_empleado', 'pais_residencia', 'sexo'])
bank_data.duplicated(subset=['ncodpers'])
#bank_data[bank_data.duplicated(subset=['ncodpers'], keep=False)]

bank_data[bank_data["ncodpers"] == 878572]