

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



train_df = pd.read_table('../input/train.tsv')

test_df = pd.read_table('../input/test.tsv')

train_df.head()
test_df.head()
size_test_df = test_df.shape
lista = np.random.randint(0 ,1000,size_test_df)
lista = lista[: , 0]
df_lista = pd.DataFrame(data = lista )

submit = pd.concat([test_df['test_id'] , df_lista] , axis = 1)

submit.columns = ['test_id' , 'price']

submit.to_csv('ramdom.csv' , index = False)
submit