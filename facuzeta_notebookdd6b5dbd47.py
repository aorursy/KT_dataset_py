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
# cargo los datos

df_training = pd.read_csv('/kaggle/input/spam-ham-nlp-mcd-udesa-2020/training.csv')

df_test = pd.read_csv('/kaggle/input/spam-ham-nlp-mcd-udesa-2020/test.csv')

df_test.head()

import re



# Voy a implementar un modelo que si el mail contiene alguna de las siguientes palabras

# entonces es spam, y sino es ham

SPAM_KEYWORDS = '(sale|free|cash)'

def clasifica(text):

    if re.search(r'(sale|free|cash)',text.lower()):

        return 'spam'

    return 'ham'



predictions = [ clasifica(t)  for t in df_test.text]



submission = pd.DataFrame(list(range(len(df_test))),columns =['Id'])

submission['Category'] = predictions

print('Tasa de spam:',(submission.Category=='spam').mean())

submission.head()
# Genero el archivo para hacer el submit

submission.to_csv('submission_clasificar_a_mano_por_palabras.csv',index=False)