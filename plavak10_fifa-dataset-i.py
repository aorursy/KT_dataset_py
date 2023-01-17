
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")


my_filepath = '../input/fifa-world-cup/WorldCups.csv'
fifa_data = pd.read_csv(my_filepath)

fifa_data.head()
plt.figure(figsize=(16,6))
sns.scatterplot(x=fifa_data['Year'], y=fifa_data['Attendance'])
plt.xlabel('Year')
plt.ylabel('Attendance')
plt.title('ATTENDANCE GROWTH OVER THE YEARS')
#Attendance_relationship with increasing years
plt.figure(figsize=(12,10))
sns.barplot(y=fifa_data['GoalsScored'], x=fifa_data['Year'])
plt.xlabel('Goals Scores')
plt.ylabel('Year')
plt.title('Goals Trend Over The Years')