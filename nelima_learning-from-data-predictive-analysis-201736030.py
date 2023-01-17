import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
input_data = pd.read_csv("../input/US_graduate_schools_admission_parameters_dataset.csv") #,index_col=0
input_data.head()