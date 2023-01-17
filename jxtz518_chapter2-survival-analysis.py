#import necessary packages
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
from lifelines.plotting import plot_lifetimes
#check all files in the floder 
print(os.listdir("../input"))
#Import the dataset
survivaldata = pd.read_csv('../input/survivalData.csv')
survivaldata.head(10) #Printing first 10 rows of the dataset
# Kaplan-Meier Estimate
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()
# T is period and E is target
T = survivaldata["daysSinceFirstPurch"]
E = survivaldata["boughtAgain"]

kmf.fit(T, event_observed=E)
#plot the result
kmf.survival_function_.plot()
plt.title('Survival function of Bought Again');
#Survival Regression
from lifelines import CoxPHFitter
#Drop gender_female, as we have another column is gender_male
survivaldata=pd.get_dummies(survivaldata).drop(columns='gender_female')
#Cox’s Proportional Hazard model
cph = CoxPHFitter()
cph.fit(survivaldata, duration_col='daysSinceFirstPurch', event_col='boughtAgain',show_progress=True)
cph.print_summary()  # access the results using cph.summary
#plot the coef
cph.plot()
# See that customer who ever returned their purchase,has a lower chance of reorder
# Note the y axis is survival rate
cph.plot_covariate_groups('returned', [0,1])
# Create a prediction dataset, we remove the "daysSinceFirstPurch"and "boughtAgain"
X = survivaldata.drop(["daysSinceFirstPurch", "boughtAgain"], axis=1)
X.head()
#predict first
cph.predict_survival_function(X)



