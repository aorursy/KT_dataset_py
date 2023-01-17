import pandas as pd 
train_data = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
train_data.describe() # checando se está tudo dentro do esperado