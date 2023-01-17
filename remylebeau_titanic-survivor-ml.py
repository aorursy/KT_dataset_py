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
x = pd.read_csv("../input/train.csv")

x_2 = pd.read_csv("../input/train.csv")

y = pd.read_csv("../input/test.csv")

toPredict = x.pop('Survived')

data = pd.concat([x,y])

data.describe(include=['O'])

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
newage = data[['Age','Pclass','Sex']].dropna()



##Getting subset data

print('Pclass 1 F = '+str(np.median(((newage.query('Pclass == 1 and Sex == "female"')))['Age'])))

print('Pclass 2 F = '+str(np.median(((newage.query('Pclass == 2 and Sex == "female"')))['Age'])))

print('Pclass 3 F = '+str(np.median(((newage.query('Pclass == 3 and Sex == "female"')))['Age'])))

print('Pclass 1 M = '+str(np.median(((newage.query('Pclass == 1 and Sex == "male"')))['Age'])))

print('Pclass 2 M = '+str(np.median(((newage.query('Pclass == 2 and Sex == "male"')))['Age'])))

print('Pclass 3 M = '+str(np.median(((newage.query('Pclass == 3 and Sex == "male"')))['Age'])))



data1 = data.query('Pclass == 1 and Sex == "female"');

data1['Age']=data1['Age'].fillna(36)