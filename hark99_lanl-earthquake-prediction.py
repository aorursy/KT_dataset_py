# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR

from sklearn.metrics import mean_absolute_error
train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train.head()
# pandas doesn't show us all the decimals

pd.options.display.precision = 15
train.head()
train.info(memory_usage='deep')
# much better!

train.head()
# Create a training file with simple derived features



rows = 50_000

segments = int(np.floor(train.shape[0] / rows))



X_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['ave', 'std', 'max', 'min','kurt','skew','time_to_failure'])

y_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['time_to_failure'])



for segment in tqdm(range(segments)):

    seg = train.iloc[segment*rows:segment*rows+rows]

    x = seg['acoustic_data'].values

    y = seg['time_to_failure'].values[-1]

    

    y_train.loc[segment, 'time_to_failure'] = y

    

    X_train.loc[segment, 'ave'] = x.mean()

    X_train.loc[segment, 'std'] = x.std()

    X_train.loc[segment, 'max'] = x.max()

    X_train.loc[segment, 'min'] = x.min()

    X_train.loc[segment, 'time_to_failure'] = y

    X_train.loc[segment, 'skew'] = ((x-x.mean())/x.std() ** 3).mean()

    X_train.loc[segment,'kurt'] = ((x-x.mean())/x.std() ** 4).mean()

    
X_train.head()
import seaborn as sns

sns.boxplot(x=X_train['ave'])
X_train['ave'].describe()
# Upper Outlier First Time

#Q1 (25%) = 4.3495

#Q3 (75%) = 4.6934

#IQR = Q3 - Q1 = 0.3439

#Outlier > Q3 + (1.5 * IQR) 

#Outlier > 4.6934 + 0.51585

#Outlier > 5.20925

# Lower Outlier

#Outlier < Q1 - (1.5* IQR)

#Outlier < 4.3495 - 0.51585

#Outlier < 3.8337



# Removing Upper Outliers

#X_train=X_train[X_train['ave']<=5.20925]

# Removing Lower Outliers

#X_train=X_train[X_train['ave']>=3.8337]



# Second Time 

# Upper Outlier

#Q1 (25%) = 4.3518

#Q3 (75%) = 4.6933

#IQR = Q3 - Q1 = 0.3415

#Outlier > Q3 + (1.5 * IQR) 

#Outlier > 4.6933 + 0.51225

#Outlier > 5.20555

# Lower Outlier

#Outlier < Q1 - (1.5* IQR)

#Outlier < 4.3518 - 0.51225

#Outlier < 3.83955



#X_train=X_train.drop([552,584,585,589,607,610,611,626,784,939])

#y_train=y_train.drop([552,584,585,589,607,610,611,626,784,939])



# Removing Upper Outliers

X_train=X_train[X_train['ave']<=5.20555]

# Removing Lower Outliers

X_train=X_train[X_train['ave']>=3.83955]
sns.boxplot(x=X_train['ave'])
sns.boxplot(x=X_train['std'])
X_train['std'].describe()
# Upper Outlier

#Q1 (25%) = 4.4741

#Q3 (75%) =6.8839

#IQR = Q3 - Q1 = 2.4098

#Outlier > Q3 + (1.5 * IQR) 

#Outlier > 6.8839 + 3.6147

#Outlier > 10.4986

#X_train=X_train[X_train['std']<=10.4986]



# Second Time

# Upper Outlier

#Q1 (25%) =  4.4370

#Q3 (75%) =6.7284

#IQR = Q3 - Q1 = 2.2914

#Outlier > Q3 + (1.5 * IQR) 

#Outlier > 6.73 + 3.4371

#Outlier > 10.1655

#X_train=X_train[X_train['std']<=10.1655]



# Third Time

# Upper Outlier

#Q1 (25%) =  4.4339

#Q3 (75%) =6.6950

#IQR = Q3 - Q1 = 2.2611

#Outlier > Q3 + (1.5 * IQR) 

#Outlier > 6.70 + 3.39165

#Outlier > 10.0867

#X_train=X_train[X_train['std']<=10.0867]



# Fourth Time

# Upper Outlier

#Q1 (25%) =  4.4331

#Q3 (75%) =6.6903

#IQR = Q3 - Q1 = 2.2572

#Outlier > Q3 + (1.5 * IQR) 

#Outlier > 6.6903 + 3.3858

#Outlier > 10.0761

#X_train=X_train[X_train['std']<=10.0761] # 10.0761



# Fifth Time

# Upper Outlier

#Q1 (25%) =  4.4330

#Q3 (75%) =6.6881

#IQR = Q3 - Q1 = 2.2551

#Outlier > Q3 + (1.5 * IQR) 

#Outlier > 6.6881 + 3.3827

#Outlier > 10.0708

X_train=X_train[X_train['std']<=10.0708]
sns.boxplot(x=X_train['std'])
X_train=X_train[X_train['std']<=9.4]
sns.boxplot(x=X_train['std'])
sns.boxplot(x=X_train['max'])
X_train['max'].describe()
# Upper Outlier

#Q1 (25%) =  90

#Q3 (75%) =162

#IQR = Q3 - Q1 = 72

#Outlier > Q3 + (1.5 * IQR) 

#Outlier > 162 + 108

#Outlier > 270

#X_train=X_train[X_train['max']<=270]



# Second Time

# Upper Outlier

#Q1 (25%) =  90

#Q3 (75%) =156

#IQR = Q3 - Q1 = 66

#Outlier > Q3 + (1.5 * IQR) 

#Outlier > 156 + 99

#Outlier > 255

#X_train=X_train[X_train['max']<=255]



# Third Time

# Upper Outlier

#Q1 (25%) =  89

#Q3 (75%) =155

#IQR = Q3 - Q1 = 66

#Outlier > Q3 + (1.5 * IQR) 

#Outlier > 155 + 99

#Outlier > 254
X_train=X_train[X_train['max']<=254]
sns.boxplot(x=X_train['max'])
X_train=X_train[X_train['max']<=231]
X_train=X_train[X_train['max']<=185]
sns.boxplot(x=X_train['max'])
sns.boxplot(x=X_train['min'])
X_train['min'].describe()
# Lower Outlier

#Q1 (25%) = -141

#Q3 (75%) = -77

#IQR = Q3 - Q1 = 64



#Outlier < Q1 - (1.5* IQR)

#Outlier < -141 - 96

#Outlier < -237

#X_train=X_train[X_train['min']>=-237]



# Second Time

# Lower Outlier

#Q1 (25%) = -138

#Q3 (75%) = -77

#IQR = Q3 - Q1 = 61



#Outlier < Q1 - (1.5* IQR)

#Outlier < -138 - 91.5

#Outlier < -229.5

#X_train=X_train[X_train['min']>=-229.5]



# Third Time

# Lower Outlier

#Q1 (25%) = -137

#Q3 (75%) = -76

#IQR = Q3 - Q1 = 61



#Outlier < Q1 - (1.5* IQR)

#Outlier < -137 - 91.5

#Outlier < -228.5
X_train=X_train[X_train['min']>=-228.5]
sns.boxplot(x=X_train['min'])
X_train=X_train[X_train['min']>=-211.5]
X_train=X_train[X_train['min']>=-175.5]
sns.boxplot(x=X_train['min'])
sns.boxplot(x=X_train['skew'])
X_train['skew'].describe()
# Upper Outlier First Time

#Q1 (25%) = -1.3

#Q3 (75%) = 1.16

#IQR = Q3 - Q1 = 2.46

#Outlier > Q3 + (1.5 * IQR) 

#Outlier > 1.16 + 3.69

#Outlier > 4.85

# Lower Outlier

#Outlier < Q1 - (1.5* IQR)

#Outlier < -1.3 - 3.69

#Outlier < -4.99



# Removing Upper Outliers

#X_train=X_train[X_train['ave']<=5.20925]

# Removing Lower Outliers

#X_train=X_train[X_train['ave']>=3.8337]



# Second Time 

# Upper Outlier

#Q1 (25%) = 4.3518

#Q3 (75%) = 4.6933

#IQR = Q3 - Q1 = 0.3415

#Outlier > Q3 + (1.5 * IQR) 

#Outlier > 4.6933 + 0.51225

#Outlier > 5.20555

# Lower Outlier

#Outlier < Q1 - (1.5* IQR)

#Outlier < 4.3518 - 0.51225

#Outlier < 3.83955



#X_train=X_train.drop([552,584,585,589,607,610,611,626,784,939])

#y_train=y_train.drop([552,584,585,589,607,610,611,626,784,939])



# Removing Upper Outliers

X_train=X_train[X_train['skew']<=3.9e-18]

# Removing Lower Outliers

X_train=X_train[X_train['skew']>=-4e-18]
sns.boxplot(x=X_train['skew'])
# Removing Upper Outliers

X_train=X_train[X_train['skew']<=3.5e-18]

# Removing Lower Outliers

X_train=X_train[X_train['skew']>=-3.5e-18]
sns.boxplot(x=X_train['skew'])
sns.boxplot(x=X_train['kurt'])
X_train['kurt'].describe()
# Upper Outlier First Time

#Q1 (25%) = -1.77

#Q3 (75%) = 1.69

#IQR = Q3 - Q1 = 3.46

#Outlier > Q3 + (1.5 * IQR) 

#Outlier > 1.69 + 4.31

#Outlier > 6

# Lower Outlier

#Outlier < Q1 - (1.5* IQR)

#Outlier < -1.77 - 3.46

#Outlier < -5.23
# Removing Upper Outliers

X_train=X_train[X_train['kurt']<=5.7e-19]

# Removing Lower Outliers

X_train=X_train[X_train['kurt']>=-5.23e-19]
sns.boxplot(x=X_train['kurt'])
sns.boxplot(x=X_train['time_to_failure'])
X_train['time_to_failure'].describe()
# Upper Outlier First Time

#Q1 (25%) = 2.33

#Q3 (75%) = 7.38

#IQR = Q3 - Q1 = 5.05 

#Outlier > Q3 + (1.5 * IQR) 

#Outlier > 7.38 + 7.56

#Outlier > 14.94
# Removing Upper Outliers

X_train=X_train[X_train['time_to_failure']<=14.92]
# Removing Upper Outliers

X_train=X_train[X_train['time_to_failure']<=14]
sns.boxplot(x=X_train['time_to_failure'])
X_train.shape
from scipy import stats



#1

pearson_coef, p_value = stats.pearsonr(X_train['ave'], X_train['time_to_failure'])

print("ave: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#2

pearson_coef, p_value = stats.pearsonr(X_train['std'], X_train['time_to_failure'])

print("std: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#3



pearson_coef, p_value = stats.pearsonr(X_train['kurt'], X_train['time_to_failure'])

print("kurt: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#4

pearson_coef, p_value = stats.pearsonr(X_train['max'], X_train['time_to_failure'])

print("max: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#5

pearson_coef, p_value = stats.pearsonr(X_train['min'], X_train['time_to_failure'])

print("min: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#6

pearson_coef, p_value = stats.pearsonr(X_train['skew'], X_train['time_to_failure'])

print("skew: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
x_train=X_train [['ave','max','min','std','kurt','skew']]

x_train.head()
x_train.shape
y_train=X_train[['time_to_failure']]
y_train.head()
y_train.shape
scaler = StandardScaler()

scaler.fit(x_train)

X_train_scaled = scaler.transform(x_train)
svm = NuSVR()

svm.fit(X_train_scaled, y_train.values.flatten())

y_pred = svm.predict(X_train_scaled)
plt.figure(figsize=(6, 6))

plt.scatter(y_train.values.flatten(), y_pred)

plt.xlim(0, 20)

plt.ylim(0, 20)

plt.xlabel('actual', fontsize=12)

plt.ylabel('predicted', fontsize=12)

plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])

plt.show()
score = mean_absolute_error(y_train.values.flatten(), y_pred)

print(f'Score: {score:0.3f}')
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
submission.head()
X_train=X_train.drop(['time_to_failure'], axis=1)
X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)
for seg_id in X_test.index:

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    x = seg['acoustic_data'].values

    

    X_test.loc[seg_id, 'ave'] = x.mean()

    X_test.loc[seg_id, 'std'] = x.std()

    X_test.loc[seg_id, 'max'] = x.max()

    X_test.loc[seg_id, 'min'] = x.min()

    X_test.loc[seg_id, 'kurt'] = ((x-x.mean())/x.std() ** 4).mean()

    X_test.loc[seg_id, 'skew'] = ((x-x.mean())/x.std() ** 3).mean()
X_test.shape
X_test_scaled = scaler.transform(X_test)

submission['time_to_failure'] = svm.predict(X_test_scaled)

submission.to_csv('submission.csv')