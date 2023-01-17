import numpy as np

import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.ensemble import AdaBoostRegressor

from sklearn.model_selection import train_test_split
path = '../input/specialists-features-normalized/'

reading_order = 'reading_order.csv'

ultrasound_info = 'ultrasound_info.csv'
df_animals = pd.read_csv(path+ultrasound_info)

df_order = pd.read_csv(path+reading_order)

order = df_order.to_numpy()
def getAOL(df, n_animal):

    #df = df.loc[df['ANIMAL'] == n_animal]["AOL (cm²)"].values

    df = df.loc[df['ANIMAL'] == n_animal]

    result = df["AOL (cm²)"].values

    return result[0]
base = ['glcm', 'glrlm', 'histogram', 'hog', 'lbp', 'resnet50', 'vgg16']

X = []

for i in range(0,7):

    X.append(np.load(path+'base/'+base[i]+'.npy'))
aol_list = [getAOL(df_animals, int(number)) for number in order]

y = np.asarray(aol_list)

y.shape
X_train = [0,0,0,0,0,0,0]

X_test = [0,0,0,0,0,0,0]

y_train = [0,0,0,0,0,0,0]

y_test = [0,0,0,0,0,0,0]



for i in range(0,7):

    X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(X[i], y, test_size=0.33, random_state=44)
print('X treino', X_train[0].shape)

print('X teste', X_test[0].shape)

print('y treino', y_train[0].shape)

print('y teste', y_test[0].shape)
y_pred = [0,0,0,0,0,0,0]

regressor = AdaBoostRegressor(random_state=0, n_estimators=200)



for i in range(0,7):

    regressor.fit(X_train[i], y_train[i])

    y_pred[i] = regressor.predict(X_test[i])

    #print(regressor.score(X_train[i], y_train[i]))
for i in range(0,7):

    print(base[i].upper()+' - Mean Squared Error:', mean_squared_error(y_test[i], y_pred[i], squared=True))

    print(base[i].upper()+' - R2 Score (Coefficient of Determination):', r2_score(y_test[i], y_pred[i]))

    print(base[i].upper()+' - Mean Absolute Error:', mean_absolute_error(y_test[i], y_pred[i]))

    print('\n')
np.round(-0.0009066444550696051, 2)