import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.head()
df.describe()
df.info()
#null data 확인

df.isna().sum()
df.iloc[:,-1].value_counts()
sns.countplot('Class', data=df)

plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
# 신용카드 사기 건수와 정상 건수의 비율 확인

df.iloc[:,-1].value_counts() / df.iloc[:,-1].count() * 100
# Seaborn을 사용한 데이터 분포 시각화

# https://datascienceschool.net/view-notebook/4c2d5ff1caab4b21a708cc662137bc65/
# 참고

# https://jaehyeongan.github.io/2018/06/30/%EC%9D%B4%EC%83%81%ED%83%90%EC%A7%80-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EC%9D%84-%ED%86%B5%ED%95%9C-%EC%9D%B4%EC%83%81%EA%B1%B0%EB%9E%98%ED%83%90%EC%A7%80-FDS/

# 시간(Time)대별 정상/부정 거래 비율

# 시간대별 트랜잭션 양

f, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(12,4))

ax1.hist(df.Time[df.Class==1], bins=50)

ax2.hist(df.Time[df.Class==0], bins=50)



ax1.set_title('Fraud')

ax2.set_title('Normal')

plt.xlabel('Time(in Seconds)'); plt.ylabel('Number of Transactions')

plt.show()
# 금액(Amount)대별 정상/부정 거래 비율

# 금액대별 트랜잭션 양

f, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(12,4))

ax1.hist(df.Amount[df.Class==1], bins=30)

ax2.hist(df.Amount[df.Class==0], bins=30)

ax1.set_title('Fraud')

ax2.set_title('Normal')



plt.xlabel('Amount ($)')

plt.ylabel('Number of Transactions')

plt.yscale('log')

plt.show()
# 비식별칼럼 정상/부정거래 비율

#Select only the anonymized features.

v_features = df.iloc[:,1:29].columns
for cnt, col in enumerate(df[v_features]):

    sns.distplot(df[col][df.Class==1], bins=50)

    sns.distplot(df[col][df.Class==0], bins=50)

    plt.legend(['Y','N'], loc='best')

    plt.title('histogram of feature '+str(col))

    plt.show()
# scikit-learn의 전처리 기능

# https://datascienceschool.net/view-notebook/f43be7d6515b48c0beb909826993c856/
# 정규화

df1=df



from sklearn.preprocessing import StandardScaler

df1['scaled_amount'] = StandardScaler().fit_transform(df1['Amount'].values.reshape(-1,1))

df1['scaled_time'] = StandardScaler().fit_transform(df1['Time'].values.reshape(-1,1))

df1.head
df2 = df1.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8', 'Time', 'Amount'], axis =1)

df2.head
df2.info()
# 1 : Isolation Forest
X1 = df2.drop('Class', axis=1)

Y1 = df2['Class']
# Under Sampling

# https://imbalanced-learn.readthedocs.io/en/stable/under_sampling.html



from imblearn.under_sampling import RandomUnderSampler

sampler = RandomUnderSampler(random_state=0)

X1, Y1 = sampler.fit_sample(X1, Y1)
from sklearn.model_selection import train_test_split



# Whole dataset

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,Y1,test_size = 0.3, random_state = 0)
from sklearn.ensemble import IsolationForest



clf = IsolationForest(n_estimators=300, contamination=0.40, random_state=42)

clf.fit(X1)

pred_outlier = clf.predict(X1)

pred_outlier = pd.DataFrame(pred_outlier).replace({1:0, -1:1})
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import itertools



class_name = [0, 1]

def plot_confusion_matrix(classes, pred, y_test1, 

                          normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    

    cm = confusion_matrix(y_test1, pred)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)

    

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")





# 평가

print('confusion matrix\n', confusion_matrix(pred_outlier, Y1))

print('Accuracy: ',accuracy_score(pred_outlier, Y1))

print('classification_report\n', classification_report(pred_outlier, Y1))

plot_confusion_matrix(class_name, pred_outlier, Y1, title='Isolation Forest')
#2 : Tensor Flow

# Credit Card Fraudulent Detection with DNN (Deep Neural Network)

# https://github.com/DakshMiglani/Credit-Card-Fraud
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
data['Class'].unique() # 0 = no fraud, 1 = fraudulent
X2 = data.iloc[:, :-1].values

Y2 = data.iloc[:, -1].values
from sklearn.model_selection import train_test_split



X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size=0.1, random_state=1)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train2 = sc.fit_transform(X_train2)

X_test2 = sc.transform(X_test2)
import keras



from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout
clf2 = Sequential([

    Dense(units=16, kernel_initializer='uniform', input_dim=30, activation='relu'),

    Dense(units=18, kernel_initializer='uniform', activation='relu'),

    Dropout(0.25),

    Dense(20, kernel_initializer='uniform', activation='relu'),

    Dense(24, kernel_initializer='uniform', activation='relu'),

    Dense(1, kernel_initializer='uniform', activation='sigmoid')

])
clf2.summary()
clf2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
clf2.fit(X_train2, Y_train2, batch_size=15, epochs=2)
score = clf2.evaluate(X_test2, Y_test2, batch_size=128)

print('\nAnd the Score is ', score[1] * 100, '%')