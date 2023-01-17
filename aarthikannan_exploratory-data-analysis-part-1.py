from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
 for filename in filenames:
  print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
print(os.listdir("../input"))
df = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv', delimiter=',')
df.shape
df.columns
df.head(10)
df.isnull().sum()
df.State.value_counts()
b = df.Year.max()
a = df.Year.min()
print("The year range from "+ str(a) +" to" + str( b))
df.Month.value_counts()
def MonthConversion(m):
    if m == 1:
        return "Jan";
    if m ==2:
        return "Feb";
    if m == 3:
        return "Mar";
    if m == 4:
        return "Apr";
    if m == 5:
        return "May";
    if m == 6:
        return "Jun";
    if m == 7:
        return "Jul";
    if m == 8:
        return "Aug";
    if m == 9:
        return "Sep";
    if m == 10:
        return "Oct";
    if m == 11:
        return "Nov";
    if m == 12:
        return "Dec";
    
df["Month"] = df["Month"].apply(MonthConversion)
df["Month"].value_counts()
df
df.Region.value_counts()
Asia_df = df[df["Region"] == "Asia"]
Asia_df
Asia_df["Country"].value_counts()
df["Month"] = df["Month"].astype(str)
df["Day"] = df["Day"].astype(str)
df["Year"] = df["Year"].astype(str)
df["Date"] = df["Month"] + df["Day"] + df["Year"]
df.Date.value_counts()
df['Year'].value_counts().sort_values()
df.Year = df.Year.astype('int64')
df.drop(df[df['Year'] < 2000].index, inplace = True) 
df['Year'].value_counts().sort_values()
df["Year"] = df["Year"].astype(str)
df["Date"] = df["Month"] + df["Day"] + df["Year"]
import matplotlib.pyplot as plt
Asia_df.head()
Asia_df.hist()

plt.show()
plt.barh(Asia_df.Country, Asia_df.Year,align = "Center")

plt.show()