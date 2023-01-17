# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

data_cereal = pd.read_csv('../input/cereal.csv')

data_cereal.describe()



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure('Bar Chart', figsize = (12, 9))

sns.countplot(y = 'calories', data = data_cereal,

             edgecolor = sns.color_palette("dark", 11)).set_title('Bar chart for calorific value across all cereals')
plt.figure(figsize = (12, 24))

sns.countplot(y = 'mfr',

              hue = 'calories',

              linewidth = 2,

              edgecolor = sns.color_palette("dark", 7),

              data = data_cereal).set_title('Calorific Values across manufacturers')


sns.factorplot(x = 'calories', 

               col = 'mfr', kind ='count',

               data= data_cereal,

               size=10, aspect=1)