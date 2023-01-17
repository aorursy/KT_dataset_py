from fastai import *

from fastai.tabular import *

import pandas as pd
path = Path('../input/credit-card/UCI_Credit_Card.csv')
df = pd.read_csv('../input/credit-card/UCI_Credit_Card.csv')

df.head()
df.isnull().sum()

df_clean = df.dropna()

df.head()
df.corr()

# joining all pay_amt column in one, aslo all bill_amt is one column

df['PAY_AMT'] = df['PAY_AMT1'] + df['PAY_AMT2']+ df['PAY_AMT4']+df['PAY_AMT5']+ df['PAY_AMT6']

df['BILL_AMT'] = df['BILL_AMT1'] +  df['BILL_AMT2']+  df['BILL_AMT3']+df['BILL_AMT4']+  df['BILL_AMT5']+  df['BILL_AMT6']
df
dep_var = 'LIMIT_BAL'

cont_names = ["SEX","EDUCATION","AGE","PAY_AMT","BILL_AMT"]

# cont_names = ["AGE","PAY_AMT","BILL_AMT"]

# procs = [FillMissing, Categorify, Normalize]
test = TabularList.from_df(df = df, path=path, cont_names=cont_names)
# data= (TabularList.from_df(df, path=path, cont_names=cont_names)

data= (TabularList.from_df(df, path=path, cont_names=cont_names, procs=procs)

# data= (TabularList.from_df(df, path=path, cont_names=cont_names)

                           .split_by_idx(list(range(800, 1000)))

                            .label_from_df(cols=dep_var)

#                            .add_test(test, label=0)

                           .databunch()

      )

                        

       

                                                       

      

      

      
data.show_batch(rows=20)
# learn = tabular_learner(data, layers=[100,50], metrics=accuracy)

learn = tabular_learner(data, layers=[200,100], metrics=mse)

learn.fit(7, 1e-6)

# learn.save('..mini-train')
learn.model_dir='/kaggle/working/'

learn.unfreeze() # must be done before calling lr_find

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=(1e-5))
learn.model_dir='/kaggle/working/'

learn.save('..credit-cards-stage1')