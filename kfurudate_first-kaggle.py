import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df_train = pd.read_csv('../input/champs-scalar-coupling/train.csv')

df_test = pd.read_csv('../input/champs-scalar-coupling/test.csv')

struectures = pd.read_csv('../input/champs-scalar-coupling/structures.csv')
sample_submission = pd.read_csv('../input/champs-scalar-coupling/sample_submission.csv')

sample_submission.head()
sample_submission.to_csv('submission.csv', index=False)
print(df_train.shape)

print(df_test.shape)

print(sample_submission.shape)
print(df_train.columns) 

print('*'* 20)

print(df_test.columns)

df_train.info()

df_test.info()
df_train.head(10)
df_test.head(10)
struectures.head(10)
df_tain_test = pd.concat([df_train, df_test], axis = 0, sort=False)

print(df_tain_test.shape)

df_tain_test.describe()
df_tain_test.describe(include='O')
from matplotlib import pyplot as plt

import seaborn as sns
sns.kdeplot(df_train.scalar_coupling_constant, shade=True)

plt.legend()

plt.show()
#df_train['log_scalar_coupling_constant'] = np.log10(df_train['scalar_coupling_constant'] +1)
#sns.kdeplot(df_train.log_scalar_coupling_constant, shade=True)

#plt.legend()

#plt.show()
plt.hist(df_train.atom_index_0, bins=12, histtype='step', normed=True, linewidth=2)

plt.hist(df_train.atom_index_1, bins=12, histtype='step', normed=True, linewidth=2)

plt.legend(['atom_index_0', 'atom_index_1'])



plt.title('atom_index Distribution')

plt.xlabel('atom_index')

plt.ylabel('Frequency')



plt.show()
train = pd.merge(

    struectures,

    df_train,  

    left_on = ['molecule_name', 'atom_index'],

    right_on= ['molecule_name', 'atom_index_0']

)



test = pd.merge(

    struectures,

    df_test,  

    left_on = ['molecule_name', 'atom_index'],

    right_on= ['molecule_name', 'atom_index_0']

)
train.head(10)
train = pd.merge(train,

                 struectures,

                 left_on=['molecule_name', 'atom_index_1'],

                 right_on=['molecule_name', 'atom_index']

                )

test = pd.merge(test,

                 struectures,

                 left_on=['molecule_name', 'atom_index_1'],

                 right_on=['molecule_name', 'atom_index']

                )
train.head()
test.head()
train = train.drop(['molecule_name', 'id', 'atom_index_x', 'atom_index_y'], axis =1)

test = test.drop(['molecule_name', 'atom_index_x', 'atom_index_y'], axis =1)
train.head(10)
def atom_number(atom):

    if atom == 'H':

        return 0

    elif atom == 'C':

        return 1

    elif atom == 'N':

        return 2

    elif atom == 'O':

        return 3

    elif atom == 'F':

        return 4
train.atom_y = [atom_number(i) for i in train.atom_y]

train.atom_x = [atom_number(i) for i in train.atom_x]

test.atom_y = [atom_number(i) for i in test.atom_y]

test.atom_x = [atom_number(i) for i in test.atom_x]
train = pd.get_dummies(train, columns=['type'], drop_first=True)

test = pd.get_dummies(test, columns=['type'], drop_first=True)
train.head(10)
train['distance'] = (

    (train['x_y'] - train['x_x'])**2 + 

    (train['y_y'] - train['y_x'])**2 + 

    (train['z_y'] - train['z_x'])**2 

) ** 0.5



test['distance'] = (

    (train['x_y'] - train['x_x'])**2 + 

    (train['y_y'] - train['y_x'])**2 + 

    (train['z_y'] - train['z_x'])**2 

) ** 0.5
train.head()
X_train = train.drop(['scalar_coupling_constant',], axis=1)

y_train = train.scalar_coupling_constant
from sklearn.model_selection import GroupKFold, train_test_split

from sklearn.metrics import accuracy_score



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train.shape, X_val.shape, y_train.shape, y_val.shape
from lightgbm import LGBMRegressor
lgb = LGBMRegressor()

lgb.fit(X_train, y_train, 

        eval_set=[(X_val, y_val)],

       early_stopping_rounds=100,

       verbose=10)
test.head()
preds = lgb.predict(X_val)
test_predictions = lgb.predict(test[['atom_x',

                                    'x_x',

                                    'y_x',

                                    'z_x',

                                    'atom_index_0',

                                    'atom_index_1',

                                    'atom_y',

                                    'x_y',

                                    'y_y',

                                    'z_y',

                                    'type_1JHN',

                                    'type_2JHC',

                                    'type_2JHH',

                                    'type_2JHN',

                                    'type_3JHC',

                                    'type_3JHH',

                                    'type_3JHN',

                                    'distance']]

                              )
sns.distplot(test_predictions)

plt.legend()

plt.show()
submission = pd.DataFrame()

submission['id'] = test['id']

submission['scalar_coupling_constant'] = test_predictions
submission.to_csv('first_sybmission.csv',index=False)