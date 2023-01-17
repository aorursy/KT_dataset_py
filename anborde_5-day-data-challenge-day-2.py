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
data = pd.read_csv("../input/train.csv", na_values=['NaN'])



data.describe()
import seaborn as sns
data = data.dropna()
sns.distplot(data['Age']).set_title("Age Histogram")
sns.distplot(data['Pclass'], kde=False).set_title("Passenger Class Histogram")
sns.distplot(data['SibSp'], kde=False).set_title("#Siblings/ Spouses Histogram")
sns.distplot(data['Parch'], kde=False).set_title("#Parents/ Children Histogram")
sns.pairplot(data, hue='Survived')