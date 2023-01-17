import numpy as np

import pandas as pd



# preparing data

responses = pd.read_csv("../input/responses.csv", index_col=False)

responses = responses.dropna()

categorical_variables = ['Gender', 

           'Left - right handed', 

           'Education', 

           'Only child', 

           'Village - town', 

           'House - block of flats']

responses = pd.get_dummies(responses, columns=categorical_variables)



list(responses.columns.values)
import seaborn as sns



sns.set(color_codes=True)



ax = sns.regplot(x="Spending on healthy eating", y="Weight", data=responses)