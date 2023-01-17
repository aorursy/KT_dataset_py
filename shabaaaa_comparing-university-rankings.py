##################################################################
## Work In Progress
##################################################################

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


from scipy.stats import friedmanchisquare
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
def removeNotNeeded(string):
    if (type(string) != str):
        return string
    else:
        return re.sub(r'\([^)]*\)|[^a-zA-Z\s0-9]+', '', string)
    
# Read the three primary datasets
shanghai = pd.read_csv('../input/shanghaiData.csv')
cwur = pd.read_csv('../input/cwurData.csv')
times = pd.read_csv('../input/timesData.csv')

# Make column names uniform all throughout
shanghai = shanghai.rename(columns = {'total_score': 'score'})
cwur = cwur.rename(columns = {'institution': 'university_name'#, 'score': 'score_cwur'
                             })
times = times.rename(columns = {'total_score': 'score'})

# Filter year variable and include data only from 2012 onwards
shang2012gre = shanghai[(shanghai.year >= 2012)]
times2012gre = times[(times.year >= 2012)]

# Get unique schools
unishang = shang2012gre.university_name.unique()
unitimes = times2012gre.university_name.unique()
unicwur = cwur.university_name.unique()

# Get common schools among the three WITHOUT cleaning university column
communi = set(unicwur).intersection(unitimes).intersection(unishang)

# Notice how some well-known schools, such as MIT, are missing from this dataset. 
# Further cleaning is needed to remedy it.
# Making a working code that will cater to a dataset with the same
# structure yet different values is what will try to achieve first.
filtshang = shang2012gre[shang2012gre['university_name'].isin(communi)]
filttimes = times2012gre[times2012gre['university_name'].isin(communi)]
filtcwur = cwur[cwur['university_name'].isin(communi)]

smolshang = filtshang[["university_name", "score", "year"]]
smoltimes = filttimes[["university_name", "score", "year"]]
smolcwur = filtcwur[["university_name", "score", "year"]]
# Year 2012 only
shang2012 = smolshang[(smolshang.year == 2012)]
times2012 = smoltimes[(smoltimes.year == 2012)]
cwur2012 = smolcwur[(smolcwur.year == 2012)]

shang2012 = shang2012[["university_name", "score"]]
times2012 = times2012[["university_name", "score"]]
cwur2012 = cwur2012[["university_name", "score"]]

shang2012['institution'] = "shanghai"
times2012['institution'] = "times"
cwur2012['institution'] = "cwur"
all2012 = pd.concat([times2012, shang2012, cwur2012], ignore_index=True)

print(all2012[:5]) #Check how the dataset looks like
all2012 = all2012.replace({'-': ''}, regex=True)
all2012[['score']] = all2012[['score']].apply(lambda x: pd.to_numeric(x, errors='ignore'))
all2012.dtypes
#print(all2012['score_times'].unique())
all2012 = all2012.fillna(0)
a2012 = all2012[all2012['score'] > 0]
# We now have a dataset that do not have unscored universities.
# We get only the universities who have ratings from the 3 institutions.
s = a2012.groupby(['university_name']).transform('count')['score']
a2012['count'] = pd.Series(s)
a2012 = a2012[ a2012['count'] == 3 ]
f, ax = plt.subplots(figsize=(8,80))
sns.barplot(y='university_name', x="score", hue="institution", data=a2012, orient='h')
# Make a dataset that can be fed to scipy.stats function friedman

shang2012 = smolshang[(smolshang.year == 2012)]
times2012 = smoltimes[(smoltimes.year == 2012)]
cwur2012 = smolcwur[(smolcwur.year == 2012)]

shang2012 = shang2012[["university_name", "score"]]
times2012 = times2012[["university_name", "score"]]
cwur2012 = cwur2012[["university_name", "score"]]

shang2012 = shang2012.rename(columns = {'score' : 'score_shang'})
cwur2012 = cwur2012.rename(columns = {'score': 'score_cwur'})                          
times2012 = times2012.rename(columns = {'score': 'score_times'})

part2012 = pd.merge(times2012, shang2012, on='university_name', how='outer')
all2012 = pd.merge(part2012, cwur2012, on='university_name', how='outer')
all2012 = all2012.replace({'-': ''}, regex=True)
all2012[['score_times']] = all2012[['score_times']].apply(lambda x: pd.to_numeric(x, errors='ignore'))
all2012 = all2012.fillna(0)
a2012 = all2012[all2012[['score_shang', 'score_times', 'score_cwur']] > 0]
a2012.dtypes
print(friedmanchisquare(a2012[1, :], a2012[2, :], a2012[3, :]))

