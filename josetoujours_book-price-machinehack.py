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
df= pd.read_excel('../input/book-price-machinehack/Data_Train.xlsx')
df_test= pd.read_excel('../input/book-price-machinehack/Data_Test.xlsx')
df= df.merge(df_test)                     
df.head()
df.shape
df.describe()
df.info()
df['Reviews'] = df.Reviews.apply(lambda r: float(r.split()[0]))
df['Ratings']= df.Ratings.str.extract('(\d+)')
df["Ratings"] = df.Ratings.astype(float)
df.head()
# Get list of categorical variables
s = (df.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)
# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: df[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])
df_1= pd.get_dummies(df, columns= ['Genre', 'BookCategory'])
df_1.head()
df_2= df_1.drop(['Title', 'Author','Synopsis','Edition'],1, inplace= False)
df_2.head()
df_2.describe()
!pip install pycaret
from pycaret.regression import *
PTA= setup(df_2, 
           target= "Price")
compare_models()
SVM= create_model('svm')
tuned_lr = tune_model(SVM)
# plot a model 
plot_model(SVM)
# predictions on hold-out set
SVM_pred_holdout = predict_model(SVM)

# evaluate a model 
evaluate_model(SVM)
# finalize model
lr_final = finalize_model(SVM)
# save a model
save_model(SVM,'SVM_model_21092020')