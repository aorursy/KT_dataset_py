import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import seaborn as sns

%pylab inline

plt.style.use('ggplot')
df = pd.read_csv('../input/cleandata.csv')

df['Age'] = (df.AgeStart + df.AgeEnd)/2

df['Height'] = (df.HeightStart + df.HeightEnd)/2

mask = df['Height'] < 200

df['Height'] = (df['Height']*mask) + (0*(~mask))

df.head()
sns.countplot(df.Gender.fillna('NA'))
sns.distplot((df.AgeStart + df.AgeEnd)/2)

plt.xlabel('Age of missing person')
plt.figure(figsize=(10, 6))

kde = True

sns.distplot(df.loc[df.Gender=='Male']['Age'].dropna(), label='Male', kde=kde)

sns.distplot(df.loc[df.Gender=='Female']['Age'].dropna(), label='Female', kde=kde)

plt.legend()
sns.distplot(df.loc[df.Gender=='Male']['Height'].dropna(), label='Male', kde=kde)

sns.distplot(df.loc[df.Gender=='Female']['Height'].dropna(), label='Female', kde=kde)

plt.legend()
plt.figure(figsize=(10, 6))

sns.countplot(df.Built)