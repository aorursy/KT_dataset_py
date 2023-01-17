import pandas as pd

pd.options.display.max_columns = 999

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline

from pylab import rcParams

rcParams['figure.figsize'] = 8, 8



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_json('../input/train.json')

test = pd.read_json('../input/test.json')

train.replace('na', np.nan, inplace=True)
i=28

if train.is_iceberg[i] == 1:

    print('iceberg')

else: print('not_iceberg')

rcParams['figure.figsize'] = 8,8  



k = 2.5



mas2 = np.array(train.band_2[i])

mas1 = np.array(train.band_1[i])



fig, (ax1, ax2) = plt.subplots(1,2)



ax1.matshow(mas1.reshape(75,75))

ax1.grid(True)

ax2.matshow(mas2.reshape(75,75))

ax2.grid(True)





fig, (ax3, ax4) = plt.subplots(1,2)



ax3.matshow(((mas1 > ((np.max(mas1)+np.min(mas1))/k).astype(int))*(-mas1)).reshape(75,75))

ax3.grid(True)

ax4.matshow(((mas2 > ((np.max(mas2)+np.min(mas2))/k).astype(int))*(-mas2)).reshape(75,75))

ax4.grid(True)





plt.show()
k = 2.5
supertrain1 = []

for i in range(train.shape[0]):  

    supertrain1.append(((mas1 > (np.max(mas1)+np.min(mas1))/k).astype(int))*(-mas1))

    

train_band_1 = pd.DataFrame(supertrain1, columns = [('('+str(i)+','+str(j)+')')for i in range(75) for j in range(75)])

train_band_1['inc_angle'] = train.inc_angle

train_band_1.inc_angle.fillna(train.inc_angle.mean(), inplace=True)
Y = train.is_iceberg
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss

from sklearn.model_selection import KFold
supertest1 = []

for i in range(test.shape[0]):

    mas1 = np.array(test.band_1[i])

    supertest1.append(((mas1 > (np.max(mas1)+np.min(mas1))/k).astype(int))*(-mas1))



test_band_1 = pd.DataFrame(supertest1, columns = [('('+str(i)+','+str(j)+')')for i in range(75) for j in range(75)])

test_band_1['inc_angle'] = test.inc_angle



model = LogisticRegression(penalty='l2', C=0.0004, random_state=100)

model.fit(train_band_1,Y)

predict = model.predict_proba(test_band_1)[:,1]

sub = pd.DataFrame({'id':test.id,'is_iceberg':predict})
sub.to_csv('sub.csv', index=False)