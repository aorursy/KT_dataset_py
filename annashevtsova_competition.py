import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV 

sample = pd.read_csv('../input/sample/sample_submission.csv')
sample.shape
line06 = pd.read_csv('../input/line-items/line06.csv')
line07 = pd.read_csv('../input/line-items/line07.csv')
shipments01 = pd.read_csv('../input/shipments/shipments2020-01-01.csv')
shipments03 = pd.read_csv('../input/shipments/shipments2020-03-01.csv')
shipments04 = pd.read_csv('../input/shipments/shipments2020-04-30.csv')
shipments06 = pd.read_csv('../input/shipments/shipments2020-06-29.csv')
sample = pd.read_csv('../input/sample/sample_submission.csv')
addresses = pd.read_csv('../input/misccc/addresses.csv')
master_categories = pd.read_csv('../input/misccc/master_categories.csv')
user_profiles = pd.read_csv('../input/misccc/user_profiles.csv')
actions = pd.read_csv('../input/messages/actions.csv')
messages = pd.read_csv('../input/messages/messages.csv')
train = pd.read_csv('../input/trains/train.csv')

shipments = pd.concat([shipments01, shipments03, shipments04, shipments06]).reset_index(drop=True)
misc = pd.concat([addresses, master_categories, user_profiles]).reset_index(drop=True)
messages = pd.concat([actions, messages]).reset_index(drop=True)
fig, axs = plt.subplots(nrows=2, figsize=(20, 20))

sns.heatmap(train_adr, ax=axs[0], annot=True, square=True, cmap='coolwarm', annot_kws={'size': 14})

for i in range(2):    
    axs[i].tick_params(axis='x', labelsize=14)
    axs[i].tick_params(axis='y', labelsize=14)
    
axs[0].set_title('Корреляции в train_adr', size=15)

plt.show()
def remove_collinear_features(x, threshold):  
    # Не убираем корреляции между фичами и целью, потому что они наоборот помогают!
    #y = x['Churn']
    #x = x.drop(columns = ['Churn'])
    
    # Считаем матрицы корреляций
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)
            
            if val >= threshold:
                drop_cols.append(col.values[0])

    # Убрать одну из коррелирующих колонок в паре
    drops = set(drop_cols)
    x = x.drop(columns = drops)
    
    # Вернуть результат обратно в датасет
    #x['Churn'] = y
               
    return x
misc.info()
train['Age'] = train['Age'].fillna( method='pad')
def display_missing(df):    
    for col in misc.columns.tolist():          
        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))
    print('\n')
    
display_missing(misc)
shipments.shape
print(shipments.info())
