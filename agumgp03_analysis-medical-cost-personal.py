import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/insurance/insurance.csv')
df.head()
df.describe()
df['age'].min(), df['age'].max()
df.isnull().sum()
df[df['sex'] == 'male']['region'].value_counts().sort_index().plot(kind='bar');
df[df['sex'] == 'female']['region'].value_counts().sort_index().plot(kind='bar');
df[(df['sex'] == 'male') & (df['smoker'] == 'yes')]['region'].value_counts().sort_index().plot(kind='bar');
df[(df['sex'] == 'female') & (df['smoker'] == 'yes')]['region'].value_counts().sort_index().plot(kind='bar');
df[df['smoker'] == 'yes']['age'].sort_index().plot(kind='hist')
alpha_color = 0.5

df[df['smoker'] == 'yes']['sex'].value_counts().sort_index().plot(kind='bar', color=['y','b'], alpha=alpha_color)
plt.hist(df.charges)



plt.title('Distribution of Charges')

plt.show()
df[df['smoker'] == 'yes']['charges'].sort_index().plot(kind='hist')
df[df['smoker'] == 'no']['charges'].sort_index().plot(kind='hist')
colors = {'yes':'r', 'no':'b'}

label = ['smoker:red']

df.plot.scatter(x='bmi', y='charges', c=df['smoker'].apply(lambda x: colors[x]), label=label, figsize=(10,8))