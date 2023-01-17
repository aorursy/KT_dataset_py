#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from pandas import Series, DataFrame
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print("Setup Complete")
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
election_type_nums = [61.27, 17.79, 13.31, 5.67, 2.62, 1.25, 0.24]

#bar plot of how many ballots cast in each election type
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.ylabel('percentage in the US')
plt.xlabel('Race/Ethnicities')
bars = ('White', 'Hispanic', 'Black', 'Asian', 'two or more races', 'American Indian', 'Native Hawaiian')
y_pos = np.arange(len(bars))
 
# Create bars
plt.bar(y_pos, election_type_nums)
 
# Create names on the x-axis
plt.xticks(y_pos, bars)

# Show graphic
plt.show()






election_type_nums = [.37, .16, .22, .0039, .2]

#bar plot of how many ballots cast in each election type
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))

plt.ylabel('Percentage of Police killings by gun 2019')
plt.xlabel('Race/Ethnicities')
bars = ('White', 'Hispanic', 'Black', 'other', 'unknown')
y_pos = np.arange(len(bars))
 
# Create bars
plt.bar(y_pos, election_type_nums)
 
# Create names on the x-axis
plt.xticks(y_pos, bars)

# Show graphic
plt.show()
election_type_nums = [37/61.27, 16/17.79, 22/13.31]

#bar plot of how many ballots cast in each election type
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.ylabel('Population size/Percentage of Police killings by gun 2019')
plt.xlabel('Race/Ethnicities')
bars = ('White', 'Hispanic', 'Black')
y_pos = np.arange(len(bars))
 
# Create bars
plt.bar(y_pos, election_type_nums)
 
# Create names on the x-axis
plt.xticks(y_pos, bars)

# Show graphic
plt.show()

