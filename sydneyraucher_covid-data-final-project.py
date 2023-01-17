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
import pandas as pd
us_counties = pd.read_csv("/kaggle/input/us-counties-covid-19-dataset/us-counties.csv")

us_counties.head()
us_counties[["cases", "deaths"]].corr()
from scipy import stats
pearson_coef, p_value = stats.pearsonr(us_counties["cases"], us_counties["deaths"])

# Pearson coefficient / correlation coefficient - how much are the two columns correlated?
print(pearson_coef)

# P-value - how sure are we about this correlation?
print(p_value)
data_honolulu = us_counties.loc[us_counties['county'] == "Honolulu"]
print(data_honolulu)
data_maricopa = us_counties.loc[us_counties['county'] == "Maricopa"]
print(data_maricopa)
data_snohomish = us_counties.loc[us_counties['county'] == "Snohomish"]
print(data_snohomish)
# libraries
import numpy as np
import matplotlib.pyplot as plt
 
# width of the bars
barWidth = 0.3
 
# Choose the height of the blue bars
bars1 = [max(data_honolulu['cases']), max(data_maricopa['cases']), max(data_snohomish['cases'])]
 
# Choose the height of the cyan bars
bars2 = [max(data_honolulu['deaths']), max(data_maricopa['deaths']), max(data_snohomish['deaths'])]
 
# Choose the height of the error bars (bars1)
yer1 = [0.5, 0.4, 0.5]
 
# Choose the height of the error bars (bars2)
yer2 = [1, 0.7, 1]
 
# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
 
# Create blue bars
plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', yerr=yer1, capsize=7, label='Cases')
 
# Create cyan bars
plt.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black', yerr=yer2, capsize=7, label='Deaths')
 
# general layout
plt.xticks([r + barWidth for r in range(len(bars1))], ['Honolulu, HI', 'Maricopa, AZ', 'Snohomish, WA'])
plt.ylabel('number of people')
plt.legend()
 
# Show graphic
plt.show()

# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
# Data
df=pd.DataFrame({'x': range(1,335), 'Honolulu': data_honolulu['cases'], 'Maricopa': data_maricopa['cases'], 'Snohomish': data_snohomish['cases'] })
 
# multiple line plot
plt.plot( 'x', 'Honolulu', data=df, marker='o', color='skyblue', linewidth=2)
plt.plot( 'x', 'Maricopa', data=df, marker='o', color='green', linewidth=2)
plt.plot( 'x', 'Snohomish', data=df, marker='o', color='purple', linewidth=2)
plt.xlabel("days since 1/21/20")
plt.ylabel("number of cases")
plt.title("Number of Cases in Honolulu, Maricopa, and Snohomish since January 21, 2020")
plt.legend()


import seaborn as sns
import matplotlib.pyplot as plt
 
# use the function regplot to make a scatterplot
sns.regplot(x=us_counties["cases"], y=us_counties["deaths"],marker="*")
plt.title("Correlation between Cases and Deaths in all US Counties")
plt.show()
 

# libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("white")
import pandas as pd
my_dpi=96
 
# Get the data (csv file is hosted on the web)
#url = 'https://python-graph-gallery.com/wp-content/uploads/gapminderData.csv'
#data = pd.read_csv('us_counties')
 
# And I need to transform my categorical column (continent) in a numerical value group1->1, group2->2...
data_honolulu = data_honolulu.loc[data_honolulu['date'] > "03-06-2020"]
frames = [data_honolulu]
us_counties = pd.concat(frames)
#us_counties = us_counties.loc[us_counties['county'] == "Snohomish" or us_counties['county'] == "Honolulu" or us_counties["county"] == "Maricopa"]
us_counties['county']=pd.Categorical(us_counties['county'])
 
# For each year:
for i in us_counties.county.unique():
 
# initialize a figure
    fig = plt.figure(figsize=(680/my_dpi, 480/my_dpi), dpi=my_dpi)
 
# Change color with c and alpha. I map the color to the X axis value.
tmp=us_counties[ us_counties.county == i ]
plt.scatter(tmp['cases'], tmp['deaths'] , s=10 , c='red', cmap="Accent", alpha=0.6, edgecolors="white", linewidth=2)

# Add titles (main and on axis)
plt.yscale('log')
plt.xlabel("cases")
plt.ylabel("deaths")
plt.title("3/6/20: "+str(i) )
plt.ylim(0,100000)
plt.xlim(30, 90)
 
# Save it
filename='Gapminder_step'+str(i)+'.png'
plt.savefig(filename, dpi=96)
plt.gca()
