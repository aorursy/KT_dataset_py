import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
            print(os.path.join(dirname, filename))
file_location = '/kaggle/input/covid19india/covidDatabaseIndia.xls'
file = pd.ExcelFile(file_location)
sheet_names = file.sheet_names
raw_data = pd.read_excel(io=file_location, sheet_name = 'Raw_Data', header = 0)
raw_data.head(5)
sns.countplot(raw_data['Current Status'])
total_cases = raw_data['Current Status'].count()
recovered = raw_data['Current Status'][raw_data['Current Status']=='Recovered'].count()
deceased = raw_data['Current Status'][raw_data['Current Status']=='Deceased'].count()
percent_recovered = recovered/total_cases
percent_deceased = deceased/total_cases
plt.title('Breakdown by patient status')
plt.show()
print('Total cases: '+str(total_cases))
print('Percent recovered: ' +str(round(percent_recovered*100,2))+'%')
print('Percent death: ' +str(round(percent_deceased*100,2))+'%')
raw_data[raw_data['Age Bracket'] == '28-35']
raw_data['Date Announced'].value_counts().plot(figsize=(18,6))
plt.title('New patients by date')
plt.show()
raw_data['Date Announced'].value_counts().sort_index().cumsum().plot(figsize=(18,6))
plt.title('Cumulative cases by date')
plt.show()
