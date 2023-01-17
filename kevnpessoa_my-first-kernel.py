# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dados = pd.read_csv('../input/train.csv')

list_dados_submissao = []



for i, row in dados.iterrows():

    if row['Sex'] == "female" or row['Pclass'] == 1 or row['Age'] < 18:

        survived = 1

    else:

        survived = 0

        

    list_dados_submissao.append({

        'PassengerId': row['PassengerId'],

        'Survived': survived

    })



df_dados_submissao = pd.DataFrame(list_dados_submissao) 