import numpy as np           
import pandas as pd          #it is used to read the csv file
import seaborn as sns
import matplotlib.pyplot as plt    #it is used for visualization 
from matplotlib import style
%matplotlib inline
data=pd.read_csv("COVID_Data.csv") #reads the covid dataset
india=data.iloc[4458:4538]   #cropping the data of India
print(india.shape)               #81 rows and 7 columns are present
india.head()            #It gives the first 5 rows

fig, ax = plt.subplots()
ax.scatter(india.Date, india.Confirmed)

# set a title and labels
ax.set_title('Corona Virus Case')
ax.set_xlabel('Days')
ax.set_ylabel('Number of cases')
