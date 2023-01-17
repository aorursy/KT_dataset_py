import numpy as np

import pandas as pd



from matplotlib import pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/infopulsehackathon/train.csv', index_col='Id')

test = pd.read_csv('../input/infopulsehackathon/test.csv', index_col='Id')



train.head()
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()

ohe_cols = train.loc[:,train.dtypes == 'object'].columns



ohe_data = pd.DataFrame(ohe.fit_transform(train[ohe_cols]).toarray(), dtype=int)

train = pd.concat([train.drop(columns = ohe_cols), ohe_data], axis=1)



ohe_data = pd.DataFrame(ohe.transform(test[ohe_cols]).toarray(), dtype=int)

X_test = pd.concat([test.drop(columns = ohe_cols), ohe_data], axis=1)



train.head()
X = train.drop(columns = 'Energy_consumption')

y = train['Energy_consumption']
from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score, KFold

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X = scaler.fit_transform(X)

X_test = scaler.transform(X_test)



kf = KFold(n_splits=5, shuffle=True, random_state=42)
from keras.layers import Input, Dense, Dropout

from keras.models import Model

from keras.optimizers import Adam



def create_model():

    inps = Input(shape=(X.shape[1],))

    x = Dense(256, activation='relu')(inps)

    x = Dropout(0.2)(x)

    x = Dense(128, activation='relu')(x)

    x = Dropout(0.1)(x)

    x = Dense(1)(x)

    model = Model(inputs=inps, outputs=x)

    model.compile(

        optimizer=Adam(lr=1e-3),

        loss=['mse']

    )

    #model.summary()

    return model
from keras import callbacks

from sklearn.metrics import mean_squared_error



test_predictions = []

metric_results = []



for ind, (tr, val) in enumerate(kf.split(X)):

    X_tr = X[tr]

    y_tr = y[tr]

    X_vl = X[val]

    y_vl = y[val]

    

    model = create_model()

    es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=50, verbose=1, mode='auto', restore_best_weights=True)

    rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, mode='auto', verbose=1)

    model.fit(

        X_tr, y_tr, epochs=500, batch_size=256, validation_data=(X_vl, y_vl), verbose=False, callbacks=[es, rlr]

    )

    test_predictions.append(model.predict(X_test).flatten())

    metric_results.append(mean_squared_error(y_vl, model.predict(X_vl).flatten()))

    print(metric_results[-1])

    

print(f'Average MSE: {np.mean(metric_results)}')
prediction = np.array(test_predictions).mean(axis=0)
print('Train target distribution')

y.hist(bins=30);



plt.show()



print('Test Prediction distribution')

pd.Series(prediction).hist(bins=30);
sub = pd.read_csv('../input/infopulsehackathon/sample_submission.csv', index_col='Id')

sub['Energy_consumption'] = prediction

sub.to_csv('submission.csv')

sub