import seaborn as sns

import pandas as pd

sns.set(style="darkgrid")

tips = pd.read_csv('../input/tips.csv')

print(tips.head())

#sns.load_dataset("tips")

g = sns.jointplot("total_bill", "tip", data=tips, kind="reg",

                  xlim=(0, 60), ylim=(0, 12), color="m", height=7)