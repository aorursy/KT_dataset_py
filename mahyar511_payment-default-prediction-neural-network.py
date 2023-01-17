# Import General Libraries 

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
# Loading Data

data = pd.read_csv('../input/UCI_Credit_Card.csv')

# Inspecting Data; Preliminary Review 

data.info()

print('------------------------')

print(data.describe())

print('------------------------')

# Credit Limit (Feature name :'LIMIT_BAL' )

%matplotlib inline

bins = 30

plt.hist(data.LIMIT_BAL, bins = bins, color='m',label = 'Total',alpha=0.5)

plt.hist(data.LIMIT_BAL[data['default.payment.next.month'] == 1], bins = bins, color='b',label = 'Default')



plt.xlabel('Credit Limit (NT dollar)');plt.ylabel('Number of Accounts')

plt.title('Fig.1 : Credit Limit ',fontweight="bold", size=12)

plt.legend();plt.show()
# Gender (Feature name :'SEX' )

men = data['SEX'][data['SEX']==1].count()

men_default = data['SEX'][(data['SEX']==1)&(data['default.payment.next.month']==1)].count()



women = data['SEX'][data['SEX']==2].count()

women_default = data['SEX'][(data['SEX']==2)&(data['default.payment.next.month']==1)].count()



total_sex = [men, women]

default_sex = [men_default,women_default]

gender = [1,2]

plt.bar(gender,total_sex, color='m',alpha=0.5, label='Total')

plt.bar(gender,default_sex, color='b',alpha=0.5, label='Default')



plt.xticks([1,2],['Male','Women']);plt.ylabel('Number of Accounts')

plt.title('Fig.2 : Gender ',fontweight="bold", size=12)

plt.legend();plt.show()
# Education (Feature name :'EDUCATION' )

grad =  data['EDUCATION'][data['EDUCATION']==1].count()

grad_default = data['EDUCATION'][(data['EDUCATION']==1)&(data['default.payment.next.month']==1)].count()



uni =  data['EDUCATION'][data['EDUCATION']==2].count()

uni_default = data['EDUCATION'][(data['EDUCATION']==2)&(data['default.payment.next.month']==1)].count()



high =  data['EDUCATION'][data['EDUCATION']==3].count()

high_default = data['EDUCATION'][(data['EDUCATION']==3)&(data['default.payment.next.month']==1)].count()



other =  data['EDUCATION'][data['EDUCATION'] > 3].count()

other_default = data['EDUCATION'][(data['EDUCATION'] > 3)&(data['default.payment.next.month']==1)].count()



total_education = [grad, uni, high, other]

default_education = [grad_default,uni_default,high_default, other_default]

degree = [1,2,3,4]

plt.bar(degree,total_education, color='m',alpha=0.5, label='Total')

plt.bar(degree,default_education, color='b',alpha=0.5, label='Default')



plt.xticks([1,2,3,4],['Grad School','University','High School','Other'])

plt.ylabel('Number of Accounts');plt.title('Fig.3 : Education ',fontweight="bold", size=12)

plt.legend();plt.show()
# Marriage Status (Feature name :'MARRIAGE')

married =  data['MARRIAGE'][data['MARRIAGE']==1].count()

married_default = data['MARRIAGE'][(data['MARRIAGE']==1)&(data['default.payment.next.month']==1)].count()



single =  data['MARRIAGE'][data['MARRIAGE']==2].count()

single_default = data['MARRIAGE'][(data['MARRIAGE']==2)&(data['default.payment.next.month']==1)].count()



other =  data['MARRIAGE'][data['MARRIAGE']==3].count()

other_default = data['MARRIAGE'][(data['MARRIAGE']==3)&(data['default.payment.next.month']==1)].count()



total_marriage = [married, single, other]

default_marriage = [married_default,single_default,other_default]

status = [1,2,3]

plt.bar(status,total_marriage, color='m',alpha=0.5, label='Total')

plt.bar(status,default_marriage, color='b',alpha=0.5, label='Default')



plt.xticks([1,2,3],['Married','Single','Other']); plt.ylabel('Number of Accounts')

plt.title('Fig.4 : Marriage Status ',fontweight="bold", size=12)

plt.legend(); plt.show()
# Age (Feature name :'AGE' )

bins = 25



plt.hist(data['AGE'],bins = bins, color='m',label = 'Total',alpha=0.5)

plt.hist(data['AGE'][data['default.payment.next.month']==1], bins = bins, color='b',label = 'Default',alpha=0.5)



plt.xlabel('Age'); plt.ylabel('Number of Accounts')

plt.title('Fig.5 : Age ',fontweight="bold", size=12)

plt.legend();plt.show()
# Payment status (Feature name :'PAY_' )

import matplotlib.gridspec as gridspec

features = list(data.columns)

plt.figure(figsize=(14,10))



gs = gridspec.GridSpec(3,2)

i = 0



plt.suptitle('Fig.6 : Payment Status',fontweight="bold", fontsize=22)

for cn in features[6:12]:

    ax = plt.subplot(gs[i])

    delay = np.zeros(12)

    delay_default = np.zeros(12)  

    for j in np.arange(0,12):

            delay[j] = data[cn][data[cn]==j-2].count()

            delay_default[j] = data[cn][(data[cn]==j-2)&(data['default.payment.next.month']==1)].count()  



    month = [-2,-1,0,1,2,3,4,5,6,7,8,9]

    plt.bar(month,delay, color='c',alpha=0.5, label='Total')

    plt.bar(month,delay_default, color='k',alpha=0.5, label='Default')



    plt.xticks([-2,-1,0,1,2,3,4,5,6,7,8,9],['0 Bill','Duly','Partial','1','2','3','4','5','6','7','8','9'], fontweight="bold", size=10)

    ax.set_xlabel('Delay (month)')

    months = ['Spetember','August','July','June','May','April']

    ax.set_title('Payment status in ' + months[i], fontweight="bold", size=12)

    ax.legend()

    i+=1

    

plt.tight_layout()

plt.subplots_adjust(top=0.90)

plt.show()
# Amount of bill statement (Feature name :'BILL_AMT_' )



features = list(data.columns)

plt.figure(figsize=(10,10))

gs = gridspec.GridSpec(3,2)

i = 0

plt.suptitle('Fig.7 : Amount of bill statement',fontweight="bold", fontsize=22)

for cn in features[12:18]:

    ax = plt.subplot(gs[i])

    bins = 25

    plt.hist(data[cn],bins = bins, color='purple',label = 'Total',alpha=0.5)

    plt.hist(data[cn][data['default.payment.next.month']==1], bins = bins, color='lightgray',label = 'Default',alpha=0.9)



    plt.xlabel('Amount (NT dollar)')

    plt.ylabel('Number of Accounts')

    ax.set_yscale('log', nonposy='clip')



    months = ['Spetember','August','July','June','May','April']

    ax.set_title('Amount of bill statement in  ' + months[i], fontweight="bold", size=12)

    ax.legend()

    i+=1  

    plt.tight_layout()

plt.subplots_adjust(top=0.90)

plt.show()
# Amount of previous payment (Feature name :'PAY_AMT_' )

features = list(data.columns)

plt.figure(figsize=(10,10))

gs = gridspec.GridSpec(3,2)

i = 0

plt.suptitle('Fig.8 : Amount of previous payment',fontweight="bold", fontsize=22)

for cn in features[18:24]:

    ax = plt.subplot(gs[i])

    bins = 25

    plt.hist(data[cn],bins = bins, color='lightblue',label = 'Total',alpha=1)

    plt.hist(data[cn][data['default.payment.next.month']==1], bins = bins, color='k',label = 'Default',alpha=0.5)



    plt.xlabel('Amount (NT dollar)')

    plt.ylabel('Number of Accounts')

    ax.set_yscale('log', nonposy='clip')

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    months = ['Spetember','August','July','June','May','April']

    ax.set_title('Amount of previous payment in  ' + months[i], fontweight="bold", size=12)

    ax.legend()

    i+=1      

plt.tight_layout()

plt.subplots_adjust(top=0.90)

plt.show()
# Making correlation coefficients pair plot of all feature in order to identify degenrate features

plt.figure(figsize=(25,25))

ax = plt.axes()

corr = data.drop(['ID'], axis=1).corr()

sns.heatmap(corr, vmax=1,vmin=-1, square=True, annot=True, cmap='Spectral',linecolor="white", linewidths=0.01, ax=ax)

ax.set_title('Fig.9 : Correlation Coefficient Pair Plot',fontweight="bold", size=30)

plt.show()
from sklearn.preprocessing import StandardScaler

from keras.utils import to_categorical



predictors = data.drop(['ID','default.payment.next.month'], axis=1).as_matrix()

predictors = StandardScaler().fit_transform(predictors)



target = to_categorical(data['default.payment.next.month']) 
# Deep Learning Libraries

from keras.layers import Dense

from keras.models import Sequential

from keras.callbacks import EarlyStopping

from sklearn.utils import class_weight
# Calculating default Ratio

non_default = len(data[data['default.payment.next.month']==0])

default = len(data[data['default.payment.next.month']==1])

ratio = float(default/(non_default+default))

print('Default Ratio :',ratio)


n_cols = predictors.shape[1]

early_stopping_monitor = EarlyStopping(patience=2)

class_weight = {0:ratio, 1:1-ratio}



model = Sequential()

model.add(Dense(25, activation='relu', input_shape = (n_cols,)))

model.add(Dense(25, activation='relu'))

#model.add(Dense(20, activation='relu'))

model.add(Dense(2, activation='softmax'))



model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(predictors, target, epochs=20, validation_split=0.3, callbacks = [early_stopping_monitor],class_weight=class_weight)
optimization = pd.DataFrame()

optimization['Hidden Layer'] = [1,1,1,2,2,3]

optimization['Node per Layer'] = [25,50,100,25,50,25]

optimization['val_loss'] = [0.1874,0.1871,0.1876,0.1861,0.1875,0.1881]



#print(optimization.to_string(index=False))

optimization.head(6)