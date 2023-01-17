import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
pd.set_option('display.max_columns',None) #To see all the columns since there are a lot of them
instruments = pd.read_pickle('../input/instruments.p')
instruments[instruments.simple_name.notna() & instruments.div_growth.notna()].head()
print(instruments.info(verbose=False))
%matplotlib inline
msno.matrix(instruments.sample(500), labels=True, sort='descending')
category_cols =  list(instruments.select_dtypes(include='category').columns)
category_cols
for a in category_cols:
    f, ax = plt.subplots(figsize=(20, 7))
    sns.countplot(x = a, data = instruments, orient='h', order = instruments[a].value_counts().iloc[:30].index)
    ax.set_title(a)
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
numeric_cols =  list(instruments.select_dtypes(include='float').columns)
numeric_cols
instruments.loc[:, numeric_cols].describe().T
