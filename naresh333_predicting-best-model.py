!pip install pycaret
import pandas as pd
from pycaret.classification import *
data = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
data.head()
Setup = setup(data= data, target='Outcome',remove_outliers=True,normalize=True)
compare_models()
lr = create_model('lr')
