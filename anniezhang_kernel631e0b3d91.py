import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 



# Input data files are available in the "../input/" directory.



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/heart.csv')
data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
data.plot(kind='scatter', x='age', y='trestbps',alpha = 0.5,color = 'red')

plt.xlabel('Age') 

plt.ylabel('Trestbps')

plt.title('Age-TrestBPS Scatter Plot') 
data.plot(kind='scatter', x='trestbps', y='chol',alpha = 0.5,color = 'red')

plt.xlabel('Resting Blood Pressure') 

plt.ylabel('Cholesterol')

plt.title('trestbps - chol Scatter Plot') 