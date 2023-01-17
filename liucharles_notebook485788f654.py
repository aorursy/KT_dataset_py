import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns
df = pd.read_csv('../input/HR_comma_sep.csv')
df.info()
df.head()
df['sales'].unique()
df['promotion_last_5years'].unique()
df['salary'].unique()
df.mean()
df.mean()['average_montly_hours']/30
print('# of people left = {}'.format(df[df['left']==1].size))

print('# of people stayed = {}'.format(df[df['left']==0].size))

print('protion of people who left in 5 years = {}%'.format(int(df[df['left']==1].size/df.size*100)))
corrmat = df.corr()

f, ax = plt.subplots(figsize=(4, 4))

# Draw the heatmap using seaborn

sns.heatmap(corrmat, vmax=.8, square=True)

plt.show()
corrmat_low = df[df['salary'] == 'low'].corr()

corrmat_medium = df[df['salary'] == 'medium'].corr()

corrmat_high = df[df['salary'] == 'high'].corr()



sns.heatmap(corrmat_low, vmax=.8, square=True,annot=True,fmt='.2f')
sns.heatmap(corrmat_medium, vmax=.8, square=True,annot=True,fmt='.2f')
sns.heatmap(corrmat_high, vmax=.8, square=True,annot=True,fmt='.2f')
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
radm = RandomForestClassifier()

radm.fit(Xtrain, ytrain)

y_val_l = radm.predict_proba(Xtest)

print("Validation accuracy: ", sum(pd.DataFrame(y_val_l).idxmax(axis=1).values

                                   == ytest)/len(ytest))
stay = df[df['left'] == 0]

stay_copy = pd.get_dummies(stay)