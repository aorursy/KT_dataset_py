!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c defense-project
!unzip defense-project.zip
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
train = pd.read_csv('train_data.csv')
scaler = StandardScaler()
test = pd.read_csv('test_data.csv')

x_train_data = train.loc[:,'# mean_0_a':'fft_749_b']
y_train_data = train.loc[:,'label']

xs_data = scaler.fit_transform(x_train_data)
test = test.to_numpy()
test = scaler.transform(test)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(xs_data, y_train_data, test_size=test_size, random_state=seed)
import xgboost as xgb
import itertools

maxdepth = list(range(5,9+1,2))
lr = list(np.arange(0.05, 0.1+0.001, 0.05))
n_est = list(range(100,1001,100))
param = list(itertools.product(lr, n_est,maxdepth))
import itertools

maxdepth = list(range(5,9+1,2))
lr = list(np.arange(0.05, 0.1+0.001, 0.05))
n_est = list(range(100,1001,100))
param = list(itertools.product(lr, n_est,maxdepth))
from tqdm import tqdm 
best = 0
best_param = None
for x in tqdm(param):
    xg_model = xgb.XGBClassifier(max_depth=x[2], learning_rate=x[0], n_estimators=x[1],seed=10024)
    xg_model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = xg_model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)
    if accuracy > best:
        best = accuracy
        best_param = x
        print(best, x)
    
xg_model = xgb.XGBClassifier(max_depth=7, learning_rate=0.1, n_estimators=100, seed=10024)
xg_model.fit(xs_data, y_train_data)
# make predictions for test data

y_pred = xg_model.predict(test)
predictions = [round(value) for value in y_pred]

real_test_df = pd.DataFrame([[i, r] for i, r in enumerate(predictions)], columns=['Id',  'Category'])
real_test_df.to_csv('result.csv', mode='w', index=False)
!kaggle competitions submit -c defense-project -f result.csv -m 'test'
!kaggle competitions submissions -c defense-project