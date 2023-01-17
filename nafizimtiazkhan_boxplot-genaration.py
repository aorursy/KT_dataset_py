import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv('../input/train-test-data/train_box_for_r2.csv')

df.rename(columns={"LR1": "L\u2081", "LR2": "L\u2082"})

df= df.replace("LR1","LR\u2081")

df= df.replace("LR2","LR\u2082")

df['accuracy']=df['accuracy']*100

sns.boxplot( x=df["algorithm"], y=df["accuracy"], width=0.8, saturation=0.85)



plt.xlabel('Models')

plt.ylabel('Cross-Validation Accuracy(%)(Training Data)')



plt.show()

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv('../input/train-test-data/test_box_for_r2.csv')

df.rename(columns={"LR1": "L\u2081", "LR2": "L\u2082"})

df= df.replace("LR1","LR\u2081")

df= df.replace("LR2","LR\u2082")

df['accuracy']=df['accuracy']*100

sns.boxplot( x=df["algorithm"], y=df["accuracy"], width=0.8, saturation=0.85)



plt.xlabel('Models')

plt.ylabel('Cross-Validation Accuracy(%)(Testing Data)')



plt.show()

# import seaborn as sns

# import matplotlib.pyplot as plt

# import pandas as pd

# df = pd.read_csv('boxplot.csv')

# df.rename(columns={"LR1": "L\u2081", "LR2": "L\u2082"})

# df= df.replace("LR1","LR\u2081")

# df= df.replace("LR2","LR\u2082")

# #df['accuracy']=df['accuracy']*100

# sns.boxplot( x=df["algorithm"], y=df["accuracy"], width=0.8, saturation=0.85)



# plt.xlabel('Algorithms')

# plt.ylabel('Accuracy of classes(%)')



# plt.show()