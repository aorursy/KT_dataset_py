import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../input/insurance.csv')
print('import done.')
df.head()
def handle_non_numeric(df):
    columns = df.columns.values
    for col in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        
        if df[col].dtype != np.int64 and df[col].dtype != np.float64:
            column_contents = df[col].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            df[col] = list(map(convert_to_int,df[col]))
    return df

df = handle_non_numeric(df)
df.head()
import seaborn as sns

Var_Corr = df.corr()
# plot the heatmap and annotation on it
plt.figure(figsize=(10,10))
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
plt.show()
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
sns.swarmplot(x='smoker', y='charges', data=df)
plt.show()