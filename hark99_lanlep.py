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

                    

                    #nrows=6000000)
# pandas doesn't show us all the decimals

pd.options.display.precision = 15
train.shape
train.info(memory_usage='deep')
train.rename({"acoustic_data": "signal", "time_to_failure": "time"}, axis="columns", inplace=True)
train.head()
# Create a training file with simple derived features



rows = 150_000

segments = int(np.floor(train.shape[0] / rows))



X_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['ave', 'std', 'max', 'min','kurt','skew','median','var','sum','q1','q19','iqr','diff'\

                                'q2','Q1','q3','q4','Q2','q6','q7','Q3','q8','q81','q82','q9','q10','abs_mean',\

                                'hmean','gmean','time_to_failure'])

y_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['time_to_failure'])



for segment in tqdm(range(segments)):

    seg = train.iloc[segment*rows:segment*rows+rows]

    x = seg['signal'].values

    y = seg['time'].values[-1]

    

    y_train.loc[segment, 'time_to_failure'] = y

    

    X_train.loc[segment, 'ave'] = np.mean(x)

    X_train.loc[segment, 'std'] = np.std(x)

    X_train.loc[segment, 'max'] = np.max(x)

    X_train.loc[segment, 'min'] = np.min(x)

    X_train.loc[segment, 'time_to_failure'] = y

    X_train.loc[segment, 'skew'] = ((x-x.mean())/x.std() ** 3).mean()

    X_train.loc[segment,'kurt'] =  ((x-x.mean())/x.std() ** 4).mean()

    X_train.loc[segment, 'sum'] = np.sum(x)

    X_train.loc[segment, 'median'] = np.median(x)

    X_train.loc[segment, 'var'] = np.var(x)

    #X_train.loc[segment, 'exp'] = np(x)

    #X_train.loc[segment, 'log'] = ln(x)

    X_train.loc[segment, 'q1'] = np.quantile(x,.1)

    X_train.loc[segment, 'q19'] = np.quantile(x,.19)

    X_train.loc[segment, 'q2'] = np.quantile(x,.2)

    X_train.loc[segment, 'Q1'] = np.quantile(x,.25)

    X_train.loc[segment, 'q3'] = np.quantile(x,.3)

    X_train.loc[segment, 'q4'] = np.quantile(x,.4)

    X_train.loc[segment, 'Q2'] = np.quantile(x,.5)

    X_train.loc[segment, 'q6'] = np.quantile(x,.6)

    X_train.loc[segment, 'q7'] = np.quantile(x,.7)

    X_train.loc[segment, 'Q3'] = np.quantile(x,.75)

    X_train.loc[segment, 'q8'] = np.quantile(x,.8)

    X_train.loc[segment, 'q81'] = np.quantile(x,.81)

    X_train.loc[segment, 'q82'] = np.quantile(x,.82)

    X_train.loc[segment, 'q9'] = np.quantile(x,.9)

    X_train.loc[segment, 'q10'] = np.quantile(x,1) #81,82

    X_train.loc[segment, 'iqr'] = np.quantile(x,.75) - np.quantile (x,.25)

    #X_train.loc[segment, 'diff'] = np.max(x) - np.min (x)

   #X_train.loc[segment, 'hmean'] = stats.hmean(x)

    #X_train.loc[segment, 'gmean'] = stats.gmean(x)

    #X_train.loc[segment, 'abs_std'] = np.abs(x).std()

    #X_train.loc[segment, 'abs_min'] = np.abs(x).min()    

    #X_train.loc[segment, 'trend'] = add_trend_feature(x)

    #X_train.loc[segment, 'abs_trend'] = add_trend_feature(x, abs_values=True)
X_train.head()
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



#7

pearson_coef, p_value = stats.pearsonr(X_train['sum'], X_train['time_to_failure'])

print("sum: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#8

pearson_coef, p_value = stats.pearsonr(X_train['median'], X_train['time_to_failure'])

print("median: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#9

pearson_coef, p_value = stats.pearsonr(X_train['var'], X_train['time_to_failure'])

print("var: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#9

pearson_coef, p_value = stats.pearsonr(X_train['gmean'], X_train['time_to_failure'])

print("gmean: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#10

pearson_coef, p_value = stats.pearsonr(X_train['q1'], X_train['time_to_failure'])

print("q1: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#11

pearson_coef, p_value = stats.pearsonr(X_train['q19'], X_train['time_to_failure'])

print("q19: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#12

pearson_coef, p_value = stats.pearsonr(X_train['q2'], X_train['time_to_failure'])

print("q2: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#13

pearson_coef, p_value = stats.pearsonr(X_train['Q1'], X_train['time_to_failure'])

print("Q1: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#14

pearson_coef, p_value = stats.pearsonr(X_train['q3'], X_train['time_to_failure'])

print("q3: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#15

pearson_coef, p_value = stats.pearsonr(X_train['q4'], X_train['time_to_failure'])

print("q4: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#16

pearson_coef, p_value = stats.pearsonr(X_train['Q2'], X_train['time_to_failure'])

print("Q2: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#17

pearson_coef, p_value = stats.pearsonr(X_train['q6'], X_train['time_to_failure'])

print("q6: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#18

pearson_coef, p_value = stats.pearsonr(X_train['q7'], X_train['time_to_failure'])

print("q7: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#19

pearson_coef, p_value = stats.pearsonr(X_train['Q3'], X_train['time_to_failure'])

print("Q3: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#20

pearson_coef, p_value = stats.pearsonr(X_train['q8'], X_train['time_to_failure'])

print("q8: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#21

pearson_coef, p_value = stats.pearsonr(X_train['q81'], X_train['time_to_failure'])

print("q81: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#22

pearson_coef, p_value = stats.pearsonr(X_train['q82'], X_train['time_to_failure'])

print("q82: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#23

pearson_coef, p_value = stats.pearsonr(X_train['q9'], X_train['time_to_failure'])

print("q9: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#24

pearson_coef, p_value = stats.pearsonr(X_train['q10'], X_train['time_to_failure'])

print("q10: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)



#25

pearson_coef, p_value = stats.pearsonr(X_train['iqr'], X_train['time_to_failure'])

print("iqr: The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
#X_train['std'].unique
#X_train['time_to_failure'].unique

X_train['time_to_failure'].describe()
#X_train=X_train[X_train['ave']<=5.20555]
# Creating a cloumn featuring binary values on the basis of accident risks

X_train['outcome'] =[1 if x<=5.682698488340905 else 0 for x in X_train['time_to_failure']]
X_train.head(2)
#x_train=X_train [['diff','ave', 'std', 'max', 'min','kurt','skew','median','var','sum','q1','q19','abs_mean',\

                               # 'q2','Q1','q3','q4','Q2','q6','q7','Q3','q8','q81','q82','q9','q10','iqr']]#=2.065

x_train=X_train [['q1','q19','q2','Q1','q3','q4','Q2','q6','q7','Q3','q8','q81','q82','q9','q10','iqr']]#=2.062

#x_train=X_train [['q1','q19','q2','q8','q81','q82','q9','min','max','std','iqr']]#=2.075

#x_train=X_train [['q1','q19','q2','q8','q81','q82','q9','iqr']] # = 2.075

x_train.head()
x_train.isnull().sum()
x_train.shape
#y_train.head()

y_train=X_train[['outcome']]

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
#X_train=X_train.drop(['time_to_failure'], axis=1)
X_test = pd.DataFrame(columns=x_train.columns, dtype=np.float64, index=submission.index)
X_test.shape
for seg_id in X_test.index:

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    x = seg['acoustic_data'].values

    

    #X_test.loc[seg_id, 'ave'] = x.mean()

    #X_test.loc[seg_id, 'std'] = np.std(x)

    #X_test.loc[seg_id, 'max'] = np.max(x)

    #X_test.loc[seg_id, 'min'] = np.min(x)

    #X_test.loc[seg_id, 'kurt'] = ((x-x.mean())/x.std() ** 4).mean()

    #X_test.loc[seg_id, 'skew'] = ((x-x.mean())/x.std() ** 3).mean()

    #X_test.loc[seg_id, 'sum'] = x.sum()

    X_test.loc[seg_id, 'q1'] = np.quantile(x,.1)

    X_test.loc[seg_id, 'q19'] = np.quantile(x,.19)

    X_test.loc[seg_id, 'q2'] = np.quantile(x,.2)

    X_test.loc[seg_id, 'Q1'] = np.quantile(x,.25)

    X_test.loc[seg_id, 'q3'] = np.quantile(x,.3)

    X_test.loc[seg_id, 'q4'] = np.quantile(x,.4)

    X_test.loc[seg_id, 'Q2'] = np.quantile(x,.5)

    X_test.loc[seg_id, 'q6'] = np.quantile(x,.6)

    X_test.loc[seg_id, 'q7'] = np.quantile(x,.7)

    X_test.loc[seg_id, 'Q3'] = np.quantile(x,.75)

    X_test.loc[seg_id, 'q8'] = np.quantile(x,.8)

    X_test.loc[seg_id, 'q81'] = np.quantile(x,.81)

    X_test.loc[seg_id, 'q82'] = np.quantile(x,.82)

    X_test.loc[seg_id, 'q9'] = np.quantile(x,.9)

    X_test.loc[seg_id, 'q10'] = np.quantile(x,1)

    X_test.loc[seg_id, 'iqr'] = np.quantile(x,.75) - np.quantile (x,.25)
X_test.shape
X_test.head()
X_test.isnull().sum()
X_test_scaled = scaler.transform(X_test)

submission['time_to_failure'] = svm.predict(X_test_scaled)

submission.to_csv('submission.csv')