# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import model_selection
import os
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

def create_folds(path, n_splits):
    df = pd.read_csv(path)
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    kf = model_selection.StratifiedKFold(n_splits=n_splits)
    for fold_, (x,y) in enumerate(kf.split(X=df, y=y)):
        df.loc[y, "kfold"] = fold_

    
    df.to_csv("/kaggle/working/train_folds.csv", index=False)

path = "/kaggle/input/siim-isic-melanoma-classification/train.csv"
n_splits = 10

if __name__ == "__main__":
    create_folds(path, n_splits)
        
