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
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 

df = pd.read_csv('../input/percentage-of-women-completing-their-degrees-in-us/percent-bachelors-degrees-women-usa.csv')
Year = df['Year']
Algriculture_percentages =  df['Agriculture']
plt.subplot(221)
plt.plot(Year,Algriculture_percentages, color = 'blue')
plt.title('Agriculture')

Architecture_percentages = df['Architecture']
plt.subplot(222)
plt.plot(Year,Architecture_percentages, color = 'red')
plt.title('Architecture')

Business = df['Business']
plt.subplot(223)
plt.plot(Year,Business, color = 'blue')
plt.title('Business')

Communications_and_Journalism = df['Communications and Journalism']
plt.subplot(224)
plt.plot(Year,Communications_and_Journalism, color = 'yellow')
plt.title('Communications and Journalism')

plt.tight_layout()
Engineering = df['Engineering']
plt.subplot(221)
plt.plot(Year,Engineering, color = 'orange')
plt.title('Engineering')

English = df['English']
plt.subplot(222)
plt.plot(Year,English, color = 'cyan')
plt.title('English')

Math_and_Statistics = df['Math and Statistics']
plt.subplot(223)
plt.plot(Year,Math_and_Statistics, color = 'black')
plt.title('Math and Statistics')

Physical_Sciences =  df['Physical Sciences']
plt.subplot(224)
plt.plot(Year,Physical_Sciences, color = 'brown')
plt.title('Physical Sciences')

plt.tight_layout()
Art_and_Performance = df['Art and Performance'] 
plt.subplot(221)
plt.plot(Year,Art_and_Performance, color = 'yellow')
plt.title('art and performance')

Biology =  df['Biology']
plt.subplot(222)
plt.plot(Year,Biology, color = 'Green')
plt.title('Biology')

Psychology =  df['Psychology']
plt.subplot(223)
plt.plot(Year,Psychology, color = 'Crimson')
plt.title('Psychology')

Public_Administration = df['Public Administration']
plt.subplot(224)
plt.plot(Year,Public_Administration, color = 'pink')
plt.title('Public Administration')

plt.tight_layout()
Foreign_Languages = df['Foreign Languages']
plt.subplot(221)
plt.plot(Year,Foreign_Languages, color = 'violet')
plt.title('Foreign Languages')

Health_Professions = df['Health Professions']
plt.subplot(222)
plt.plot(Year,Health_Professions, color = 'purple')
plt.title('Health Professions')

Computer_Science =  df['Computer Science']
plt.subplot(223)
plt.plot(Year,Computer_Science, color = 'green')
plt.title('Computer Science')

Education = df['Education']
plt.subplot(224)
plt.plot(Year,Education, color = 'red')
plt.title('Education')

plt.tight_layout()
print('median percentage of women in us joining computer science from 1970 to 2010', np.median(df['Computer Science']))
print('median percentage of women in us joining Engineering from 1970 to 2010', np.median(df['Engineering']))
print('median percentage of women in us joining Agriculture from 1970 to 2010', np.median(df['Agriculture']))
print('median percentage of women in us joining Architecture from 1970 to 2010', np.median(df['Architecture']))
print('median percentage of women in us joining Psychology from 1970 to 2010', np.median(df['Psychology']))
print('median percentage of women in us joining Foreign Languages from 1970 to 2010', np.median(df['Foreign Languages']))
print('median percentage of women in us joining Health Professions from 1970 to 2010', np.median(df['Health Professions']))
print('median percentage of women in us joining Public Administration from 1970 to 2010', np.median(df['Public Administration']))
print('median percentage of women in us joining Biology from 1970 to 2010', np.median(df['Biology']))
print('median percentage of women in us joining Business from 1970 to 2010', np.median(df['Business']))
print('median percentage of women in us joining Art and Performance from 1970 to 2010', np.median(df['Art and Performance']))
print('median percentage of women in us joining Physical Sciences from 1970 to 2010', np.median(df['Physical Sciences']))
print('median percentage of women in us joining Communications and Journalism from 1970 to 2010', np.median(df['Communications and Journalism']))
print('median percentage of women in us joining English from 1970 to 2010', np.median(df['English']))
print('median percentage of women in us joining Education from 1970 to 2010', np.median(df['Education']))
print('median percentage of women in us joining Math and Statistics from 1970 to 2010', np.median(df['Math and Statistics']))
print('Average percentage of women in us joining computer science from 1970 to 2010', np.mean(df['Computer Science']))
print('Average percentage of women in us joining Engineering from 1970 to 2010', np.mean(df['Engineering']))
print('Average percentage of women in us joining Agriculture from 1970 to 2010', np.mean(df['Agriculture']))
print('Average percentage of women in us joining Architecture from 1970 to 2010', np.mean(df['Architecture']))
print('Average percentage of women in us joining Psychology from 1970 to 2010', np.mean(df['Psychology']))
print('Average percentage of women in us joining Foreign Languages from 1970 to 2010', np.mean(df['Foreign Languages']))
print('Average percentage of women in us joining Health Professions from 1970 to 2010', np.mean(df['Health Professions']))
print('Average percentage of women in us joining Public Administration from 1970 to 2010', np.mean(df['Public Administration']))
print('Average percentage of women in us joining Biology from 1970 to 2010', np.mean(df['Biology']))
print('Average percentage of women in us joining Business from 1970 to 2010', np.mean(df['Business']))
print('Average percentage of women in us joining Art and Performance from 1970 to 2010', np.mean(df['Art and Performance']))
print('Average percentage of women in us joining Physical Sciences from 1970 to 2010', np.mean(df['Physical Sciences']))
print('Average percentage of women in us joining Communications and Journalism from 1970 to 2010', np.mean(df['Communications and Journalism']))
print('Average percentage of women in us joining English from 1970 to 2010', np.mean(df['English']))
print('Average percentage of women in us joining Education from 1970 to 2010', np.mean(df['Education']))
print('Average percentage of women in us joining Math and Statistics from 1970 to 2010', np.mean(df['Math and Statistics']))