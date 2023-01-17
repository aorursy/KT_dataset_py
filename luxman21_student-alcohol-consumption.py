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
# Import the dependecies 

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

from sklearn import preprocessing

data = pd.read_csv('/kaggle/input/student-alcohol-consumption/student-por.csv')

display(data)
# get all the catagoricla fields 

categorical_Fields = data.select_dtypes(include=[object])

print(categorical_Fields.columns)

categorical_Fields.head(5)

print(type(categorical_Fields))



#checking for nulls 

categorical_Fields[categorical_Fields.isnull().any(axis=1)]
# encode labels with value between 0 and n_classes-1.

le = preprocessing.LabelEncoder()



columnsThatNeedsToBeConverted = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',

       'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',

       'nursery', 'higher', 'internet', 'romantic']



finalDataSet = data





#print(finalDataSet['{}'.format('school')])

#print(type(finalDataSet))





for col in columnsThatNeedsToBeConverted: 

    finalDataSet["{}".format(col)] = finalDataSet["{}".format(col)].astype('category')

    finalDataSet["{}".format(col)] = finalDataSet["{}".format(col)].cat.codes

#    finalDataSet = finalDataSet.apply(le.fit_transform(data['{}'.format(col)]))



finalDataSet.head(10)
temp = finalDataSet[['school', 'freetime']]

sns.pairplot(temp);

#sm = pd.plotting.scatter_matrix(finalDataSet, alpha=0.4, figsize=((10,10)))
# plotting a heat map 

plt.figure(figsize=(20,20))

sns.heatmap(finalDataSet.corr(),annot = True,fmt = ".2f",cbar = True)

plt.xticks(rotation=90)

plt.yticks(rotation = 0)
ages = finalDataSet["age"].value_counts()

print(ages)

labels = (np.array(ages.index))

print(labels)

sizes = (np.array((ages / ages.sum())*100))







fig1, ax1 = plt.subplots(figsize=(8, 7))

colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#deda00','#d9291c','#deda00','#00de0f']

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',colors = colors,

        shadow=False, startangle=0)



ax1.plot(figsize=(50,50))



# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.show()