import numpy as np 

import pandas as pd 

from pathlib import Path

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline
np.random.seed(32)
data_dir = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray')



# Path to train directory

train_dir = data_dir / 'train'



# Path to validation directory

val_dir = data_dir / 'val'



# Path to test directory

test_dir = data_dir / 'test'
normal_dir = train_dir / 'NORMAL'

pneumonia_dir = train_dir / 'PNEUMONIA'



# Get the list of all the images

normal = normal_dir.glob('*.jpeg')

pneumonia = pneumonia_dir.glob('*.jpeg')



# Empty list

train_data = []



# Label all Normal cases as 0

for img in normal:

    train_data.append((img,0))



# Label all Pneumonia cases as 1

for img in pneumonia:

    train_data.append((img, 1))



# Data in list converted to pandas DataFrame

train_data = pd.DataFrame(train_data, columns=['image', 'label'],index=None)
#Viewing random 5 examples of the dataset



train_data.sample(5)
#Getting count for both Normal and Pneumonia cases



case_count = train_data["label"].value_counts()

print(case_count)
#Plotting the cases count



plt.figure(figsize = (6,4))

sns.set_style("whitegrid")

sns.barplot(x =['Pneumonia(1)','Normal(0)'], y = case_count)

plt.title('Number of cases', fontsize=14, weight = "bold")

plt.xlabel('Case type', fontsize=12, weight = "bold")

plt.ylabel('Count', fontsize=12, weight = "bold")

plt.show();