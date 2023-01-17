

import numpy as np # linear algebra




import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
df=pd.read_csv("../input/creditcardfraud/creditcard.csv")
df.head()
!pip install pycaret
from pycaret.classification import *
clf=setup(df,target="Class")
compare_models()
et=create_model("et")
print(et)
plot_model(et, plot = 'auc')
plot_model(et, plot = 'pr')
plot_model(et, plot='feature')
plot_model(et, plot = 'confusion_matrix')
evaluate_model(et)
predict_model(et);