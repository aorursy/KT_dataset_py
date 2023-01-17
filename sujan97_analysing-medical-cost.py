import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('../input/insurance/insurance.csv')

df.head()
plt.title('Relation between Age and Charges')

sns.scatterplot(x=df['age'],y=df['charges'])

plt.show()

plt.title('Regression between Age and Charges')

sns.regplot(x=df['age'],y=df['charges'])

plt.show()
plt.title('Relation between BMI and Charges')

sns.scatterplot(x=df['bmi'],y=df['charges'])

plt.show()

plt.title('Relation between BMI and Charges')

sns.regplot(x=df['bmi'],y=df['charges'])

plt.show()
sns.scatterplot(x=df['bmi'], y=df['charges'], hue=df['smoker'])
sns.lmplot(x="bmi", y="charges", hue="smoker", data=df)
sns.swarmplot(x=df['smoker'],y=df['charges'])
plt.figure(figsize=(14,6))

plt.title('Relation between Age and Charges')

#sns.regplot(x=df['children'],y=df['charges'])

sns.barplot(x=df['children'], y=df['charges'])
sns.swarmplot(x=df['sex'],y=df['charges'])

plt.show()

sns.scatterplot(x=df['bmi'], y=df['charges'], hue=df['sex'])

plt.show()

sns.barplot(x=df['sex'], y=df['charges'])

plt.show()
sns.swarmplot(x=df['region'],y=df['charges'])

plt.show()

sns.barplot(x=df['region'], y=df['charges'])

plt.show()