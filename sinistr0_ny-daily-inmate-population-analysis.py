import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20, 20]
%matplotlib inline
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv('../input/daily-inmates-in-custody.csv')
#clean data by removing rows containing NaN  
data = data[pd.notnull(data['RACE'])]
data.info()
plt.figure(figsize=(20,7))
plt.hist(pd.to_numeric(data['AGE']), facecolor='black', bins=100)
plt.title("Distribution of Ages")
plt.xlabel("Age of Inmates")
plt.ylabel("Count")
plt.show()

data['Date_reviewed'] = pd.to_datetime(data['ADMITTED_DT'])

plt.figure(figsize=(20,7))
plt.hist(data['Date_reviewed'].dt.hour,
             alpha=.8, bins=100)
plt.title('Arrests Made Every Hour')
plt.xlabel('hour of day')
plt.ylabel('number of records')
plt.show()
plt.figure(figsize=(20,7))
sns.countplot(x=data['Date_reviewed'].dt.hour, hue=data['RACE'], palette="Set2")
plt.title('Relationship of Race/Hour')
plt.xlabel('Hour of Day')
plt.ylabel("Number of Records")
plt.show()