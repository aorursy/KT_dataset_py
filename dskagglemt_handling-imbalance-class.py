import numpy as np  

import pandas as pd  



import matplotlib.pyplot as plt

import seaborn as sns





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
LABELS = ["No Claim Filed", "Claim Filed"]
data = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/train.csv')
data.head()
data.shape
data.target.value_counts()
sns.countplot(data.target);

plt.xlabel('Is Filed Claim?');

plt.ylabel('Number of occurrences');

plt.show()
count_classes = pd.value_counts(data['target'], sort = True)



count_classes.plot(kind = 'bar', rot = 0)



plt.title("Claims Distribution")



plt.xticks(range(2), LABELS)



plt.xlabel("Claims --> ")



plt.ylabel("Frequency --> ")



plt.show()
X = data.drop('target', axis = 1)

y = data.target



data.shape, X.shape, y.shape
# y_us.value_counts()[0] 

# X.shape[0]
data_table = pd.DataFrame()



data_table['technique'] = ['Original Data']

data_table['X_Shape'] = [X.shape[0]]

data_table['y_Shape'] = [y.shape[0]]

data_table['target_0'] = [y.value_counts()[0]]

data_table['target_1'] = [y.value_counts()[1]]



data_table
from imblearn.under_sampling import NearMiss
nm = NearMiss()
X_us, y_us = nm.fit_sample(X, y)
print('Shape for Imbalanced Class :')

display(X.shape, y.shape)

print('Count of target : {} '.format(y.value_counts()))





print('Shape for Balanced Class :')

display(X_us.shape, y_us.shape)

print('Count of target : {} '.format(y_us.value_counts()))
new_row = {'technique': 'Under Sampling - NearMiss', 'X_Shape': X_us.shape[0], 'y_Shape':y_us.shape[0], 'target_0': y_us.value_counts()[0], 'target_1' : y_us.value_counts()[1]}

data_table = data_table.append(new_row,ignore_index=True)



data_table
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42, replacement=True)  

X_rus, y_rus = rus.fit_resample(X, y)
new_row = {

    'technique': 'Under Sampling - RandomUnderSampler', 

    'X_Shape': X_rus.shape[0], 

    'y_Shape':y_rus.shape[0], 

    'target_0': y_rus.value_counts()[0], 

    'target_1' : y_rus.value_counts()[1]

}



data_table = data_table.append(new_row,ignore_index=True)



data_table
from imblearn.over_sampling import RandomOverSampler
os = RandomOverSampler() # Default sampling_strategy='auto'
X_os, y_os = os.fit_sample(X, y)
new_row = {

    'technique': 'Over Sampling - Auto', 

    'X_Shape': X_os.shape[0], 

    'y_Shape':y_os.shape[0], 

    'target_0': y_os.value_counts()[0], 

    'target_1' : y_os.value_counts()[1]

}

data_table = data_table.append(new_row,ignore_index=True)



data_table
os2 = RandomOverSampler(sampling_strategy=0.5)



X_os2, y_os2 = os2.fit_sample(X, y)
new_row = {

    'technique': 'Over Sampling - half', 

    'X_Shape': X_os2.shape[0], 

    'y_Shape':y_os2.shape[0], 

    'target_0': y_os2.value_counts()[0], 

    'target_1' : y_os2.value_counts()[1]

}

data_table = data_table.append(new_row,ignore_index=True)



data_table
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy = 'minority')

X_smote, y_smote = smote.fit_sample(X, y)
new_row = {

    'technique': 'SMOTE - minority', 

    'X_Shape': X_smote.shape[0], 

    'y_Shape':y_smote.shape[0], 

    'target_0': y_smote.value_counts()[0], 

    'target_1' : y_smote.value_counts()[1]

}

data_table = data_table.append(new_row,ignore_index=True)



data_table
from imblearn.combine import SMOTETomek
smk = SMOTETomek(random_state=9)

X_smk, y_smk = smk.fit_sample(X, y)
new_row = {

    'technique': 'SMOTETomek_9', 

    'X_Shape': X_smk.shape[0], 

    'y_Shape':y_smk.shape[0], 

    'target_0': y_smk.value_counts()[0], 

    'target_1' : y_smk.value_counts()[1]

}

data_table = data_table.append(new_row,ignore_index=True)



data_table