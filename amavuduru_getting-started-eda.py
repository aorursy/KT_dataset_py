# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/training_variants')

test = pd.read_csv('../input/test_variants')

trainx = pd.read_csv('../input/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

testx = pd.read_csv('../input/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
train.head()
trainx.head()
train_data = pd.concat([trainx, train.drop('ID', axis=1)], axis=1)

train_data.head()
plt.figure(figsize=(12,8))

sns.countplot(x="Class", data=train_data)

plt.ylabel('Frequency', fontsize=12)

plt.xlabel('Class Count', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of Classes", fontsize=15)
train_data['Gene'].nunique() #This will give us the number of unique genes.
train_data['Variation'].nunique() #This will give us the number of unique variations


train_data[train_data['Class'] == class_label]['Gene'].nunique()
classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]

for class_label in classes:

    print('Number of unique genes for class {0}: {1}'.format(class_label, train_data[train_data['Class'] == class_label]['Gene'].nunique()))
classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]

for class_label in classes:

    print('Number of unique variations for class {0}: {1}'.format(class_label, train_data[train_data['Class'] == class_label]['Variation'].nunique()))