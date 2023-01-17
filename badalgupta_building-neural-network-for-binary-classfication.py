# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
train=pd.read_csv('../input/criminal_train.csv')
test=pd.read_csv('../input/criminal_test.csv')
y=train['Criminal']
train.drop(['Criminal','PERID'],axis=1,inplace=True)
test_id=test['PERID']
test.drop('PERID',axis=1,inplace=True)

criminal_features=['IFATHER', 'NRCH17_2', 'IRHHSIZ2', 'IRKI17_2',
        'IRHH65_2',   'PRXYDATA', 'MEDICARE',
       'CAIDCHIP', 'CHAMPUS', 'PRVHLTIN', 'GRPHLTIN', 'HLTINNOS', 'HLCNOTYR',
       'HLCNOTMO', 'HLCLAST', 'HLLOSRSN', 'HLNVCOST', 'HLNVOFFR', 'HLNVREF',
       'HLNVNEED', 'HLNVSOR', 'IRMCDCHP', 'IIMCDCHP', 'IRMEDICR', 'IIMEDICR',
       'IRCHMPUS', 'IICHMPUS', 'IRPRVHLT', 'IIPRVHLT', 'IROTHHLT', 'IIOTHHLT',
       'HLCALLFG', 'HLCALL99', 'ANYHLTI2', 'IRINSUR4', 'IIINSUR4', 'OTHINS',
        'CELLWRKNG','IIFAMSOC', 
       'IIFAMSSI',  'IIFSTAMP',  'IIFAMPMT', 
       'IIFAMSVC',  'IRPINC3', 'IRFAMIN3', 'IIPINC3',
       'IIFAMIN3', 'POVERTY3',  'PDEN10',
       'COUTYP2',  'ANALWT_C']
train=train[criminal_features]
test=test[criminal_features]
train.head()
train.info()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train = sc.fit_transform(train)
test = sc.transform(test)
classifier=Sequential()

#First Layer
classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu', input_dim = 51))
#second Layer
classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu'))

#OutPUt Layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(train, y, batch_size = 10, nb_epoch = 50)
pred=classifier.predict(test)
def fun(val):
    if val > 0.5:
        return 1
    else:
        return 0
sub=pd.DataFrame(data=[],columns=['PERID','Criminal'])
sub.PERID=test_id
sub.Criminal=pred
sub['Criminal']=sub['Criminal'].apply(fun)
sub.to_csv('submission__1.csv',sep=',',index=False)
