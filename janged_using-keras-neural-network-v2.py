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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import RepeatedKFold

from keras import models

from keras import layers

from keras import optimizers

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint

from keras import backend as K
debug = False

eda_view = True



num_rows = 2000 if debug else None

train = pd.read_csv('../input/train.csv', nrows = num_rows)

test = pd.read_csv('../input/test.csv', nrows = num_rows)
for df in [train, test]:

    df['sale_yr'] = pd.to_numeric(df.date.str.slice(0, 4))

    df['sale_month'] = pd.to_numeric(df.date.str.slice(4, 6))

    df['sale_day'] = pd.to_numeric(df.date.str.slice(6, 8))

    df.drop(['date'], axis=1, inplace=True)
if eda_view == True:

    sns.distplot(train['price'])

    plt.show()

    print(train['price'].skew())

    print(train['price'].kurt())
if eda_view == True:

    for c in train:

        sns.kdeplot(train[c])

        plt.show()
features = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']
for df in [train, test]:

    for c in features:

        df[c] = np.log1p(df[c])



train['price'] = np.log1p(train['price'])
for df in [train, test]:

	df['total_rooms'] = df['bathrooms'] + df['bedrooms']

	df['total_sqft'] = df['sqft_living'] + df['sqft_lot']

	df['total_sqft15'] = df['sqft_living15'] + df['sqft_lot15']

	df['grade_multi_cond'] = df['grade'] * df['condition']

	# 용적률 주거 공간 / 대지 면적

	df['far1'] = df['sqft_living'] / df['sqft_lot']

	df['far2'] = df['sqft_living15'] / df['sqft_lot15']

    # 용적률이 늘어 났는가?

	df['is_far_incre'] = df['far2'] - df['far1']

	df['is_far_incre'] = df['is_far_incre'].apply(lambda x : 0 if x < 0 else 1)

	

    #건폐률 (토지 면적 대비 1층이 차지하는 비)

	df['far3'] = (df['sqft_above'] / df['floors']) / df['sqft_lot']

	

    # 층당 화장실 개수

	df['bath_per_floors'] = df['bathrooms'] / df['floors']

    

    # 층당 방 수

	df['bed_per_floors'] = df['bedrooms'] / df['floors']

    

    # 지상 거주 공간 중 1층이 차지하는 비

	df['1st_living'] = df['sqft_above'] / df['floors']

	df['is_renovated'] = df['yr_renovated'] - df['yr_built']

	df['is_renovated'] = df['is_renovated'].apply(lambda x: 0 if x < 0 else 1)

	df['age_built'] = df['yr_built'] / 2015

	df['age_reno'] = df['yr_renovated'] / 2015

	# 지하가 있냐 없냐?

	df['is_basement'] = df['sqft_basement'].apply(lambda x: 0 if x == 0 else 1)

    #전체 방수 + 지하실

	df['total_rooms'] = df['total_rooms'] + df['is_basement']

    # 다락이 있냐 없냐?

	df['is_top'] = df['floors'].apply(lambda x: 0 if int(x) == x else 1)

    # 2015년에 주거 공간이 커졌는가?

	df['is_living_wider'] =  df['sqft_living15'] / df['sqft_living']

	df['is_living_wider'] = df['is_living_wider'].apply(lambda x:0 if x < 1 else 1)

    # 2015년 토지 면적이 커녔는가?

	df['is_lot_wider'] =  df['sqft_lot15'] / df['sqft_lot']

	df['is_lot_wider'] = df['is_lot_wider'].apply(lambda x: 0 if x < 1 else 1)

	df['total_grade'] = df['waterfront'] + df['view'] + df['condition'] + df['grade']

train['per_price_living'] = train['price']/train['sqft_living']

train['per_price_above'] = train['price']/train['sqft_above']

train['per_price_lot'] = train['price']/train['sqft_lot']



train['per_price_living_grade'] = train['per_price_living'] * train['grade']

train['per_price_above_grade'] = train['per_price_above'] * train['grade']

train['per_price_lot_grade'] = train['per_price_lot'] / train['grade']



zipcode_price_living = train.groupby(['zipcode'])['per_price_living'].agg({'mean','median'}).reset_index()

train = pd.merge(train,zipcode_price_living, how='left',on='zipcode')

test = pd.merge(test,zipcode_price_living, how='left',on='zipcode')



zipcode_price_above = train.groupby(['zipcode'])['per_price_above'].agg({'mean','median'}).reset_index()

train = pd.merge(train,zipcode_price_above,how='left',on='zipcode')

test = pd.merge(test,zipcode_price_above,how='left',on='zipcode')



zipcode_price_lot = train.groupby(['zipcode'])['per_price_lot_grade'].agg({'mean','median'}).reset_index()

train = pd.merge(train,zipcode_price_lot,how='left',on='zipcode')

test = pd.merge(test,zipcode_price_lot,how='left',on='zipcode')



zipcode_price_living_grade = train.groupby(['zipcode'])['per_price_living_grade'].agg({'mean','median'}).reset_index()

train = pd.merge(train,zipcode_price_living_grade, how='left',on='zipcode')

test = pd.merge(test,zipcode_price_living_grade, how='left',on='zipcode')



zipcode_price_above_grade = train.groupby(['zipcode'])['per_price_above_grade'].agg({'mean','median'}).reset_index()

train = pd.merge(train,zipcode_price_above_grade,how='left',on='zipcode')

test = pd.merge(test,zipcode_price_above_grade,how='left',on='zipcode')



zipcode_price_lot_grade = train.groupby(['zipcode'])['per_price_lot_grade'].agg({'mean','median'}).reset_index()

train = pd.merge(train,zipcode_price_lot_grade,how='left',on='zipcode')

test = pd.merge(test,zipcode_price_lot_grade,how='left',on='zipcode')



train.drop(['per_price_living', 'per_price_above', 'per_price_lot', 'per_price_living_grade', 'per_price_above_grade', 'per_price_lot_grade'], axis=1, inplace=True)
if eda_view == True:

    corrmat = train.corr()

    top_corr_features = corrmat.index[abs(corrmat['price']) >= 0.45]

    plt.figure(figsize=(12, 12))

    hm = sns.heatmap(train[top_corr_features].corr(), annot=True)

    plt.show()
all_features = True

if all_features == True:

	x_train = train.drop(['price', 'id'], axis=1)

	x_test = test.drop(['id'], axis=1)

	y_train = train['price']

	y_train = y_train.values.reshape(-1,1)

else:

	x_train = train[top_corr_features]

	x_train = x_train.drop(['price'], axis=1)

	y_train = train['price']

	y_train = y_train.values.reshape(-1,1)

	features = [c for c in top_corr_features if c not in ['price']]

	x_test = test[features]
x_scaler = StandardScaler().fit(x_train)

y_scaler = StandardScaler().fit(y_train)

x_train = x_scaler.transform(x_train)

x_test = x_scaler.transform(x_test)

y_train = y_scaler.transform(y_train)
x_train=pd.DataFrame(x_train)

x_test=pd.DataFrame(x_test)



x_train = x_train.values.reshape(-1, x_train.shape[1], 1)

x_test = x_test.values.reshape(-1, x_train.shape[1], 1)
def coeff_determination(y_true, y_pred):

    SS_res =  K.sum(K.square( y_true-y_pred ))

    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )

    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
def make_model2():

    model = models.Sequential()

    model.add(layers.Dense(8, activation='relu',	input_shape=(x_train.shape[1],1)))

    model.add(layers.Dense(8, activation='relu'))

    model.add(layers.Flatten())

    #model.add(layers.Dropout(0.15))

    model.add(layers.Dense(1))



    optimizer = optimizers.RMSprop(lr=0.001)

    model.compile(optimizer= optimizer, loss = 'mse', metrics=['mae','mse', coeff_determination])

    return model
patient = 40

callbacks2 = [

    EarlyStopping(monitor='val_mean_squared_error', patience=patient, mode='min', verbose=0),

 	ReduceLROnPlateau(monitor = 'val_mean_squared_error', factor = 0.5, patience = patient / 2, min_lr=0.0001, verbose=0, mode='min'),

    #ModelCheckpoint(filepath=model_path2, monitor='val_mean_squared_error', verbose=0, save_best_only=True, mode='min'),

    #TensorBoard('./log_dir', histogram_freq=0, write_graph=True, write_images=True)

    ]
# k-fold cross validation 검증

# fix random seed for reproducibility

seed = 2019

np.random.seed(seed)
kfold = RepeatedKFold(n_splits=4, n_repeats=2, random_state=seed)

cvscores2 = []

epochs_num = 600

batch_num = 32
i = 1

for train, test in kfold.split(x_train, y_train):

    print('\n Fold : ', i)

    i = i + 1



    model2 = make_model2()

        # Fit the model

    # 여기서 사용되는 x_train[train], y_train[train] 은 3개 폴드의 매 폴드의 train과 target임

  

    model2.fit(

        x_train[train], y_train[train],

        validation_data=(x_train[test], y_train[test]),

        epochs=epochs_num, batch_size=batch_num, 

        verbose=0, 

        callbacks=callbacks2)



    # evaluate the model

    scores2 = model2.evaluate(x_train[test], y_train[test], verbose=0)

    print("\nmodel2 val_%s: %.2f%%" % (model2.metrics_names[2], scores2[2]*100))

    cvscores2.append(scores2[2] * 100)
print("\nmodel2 cvscore %.2f%% (+/- %.2f%%)" % (np.mean(cvscores2), np.std(cvscores2)))



model2 = make_model2()
patient = 30

callbacks3 = [

    EarlyStopping(monitor='mean_squared_error', patience=patient, mode='min', verbose=0),

 	ReduceLROnPlateau(monitor = 'mean_squared_error', factor = 0.5, patience = patient / 2, min_lr=0.0001, verbose=0, mode='min'),

    #ModelCheckpoint(filepath=model_path3, monitor='mean_squared_error', verbose=0, save_best_only=True, mode='min'),

    #TensorBoard('./log_dir', histogram_freq=0, write_graph=True, write_images=True)

    ]
model2.fit(

    x_train, y_train,

    epochs=300, batch_size=batch_num, 

    verbose=0,

    callbacks=callbacks3 

    )
y_preds2 = model2.predict(x_test)



inv_y_preds2 =  np.expm1(y_scaler.inverse_transform(y_preds2))
if debug == False:

    sub = pd.read_csv('../input/sample_submission.csv')

    sub['price'] = inv_y_preds2

    sub.to_csv('./keras_model2.csv', index=False)