# import necessary packages
import pandas as pd
from numpy import log
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
%matplotlib inline
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train_y = df_train['label']
df_train_x = df_train.drop('label', axis=1)
df_train_x.iloc[0].shape
plt.matshow(df_train_x.iloc[0].values.reshape(28,28))
print('Correct number: {}'.format(df_train_y.iloc[0]))
plt.matshow(df_train_x.iloc[20].values.reshape(28,28))
print('Correct number: {}'.format(df_train_y.iloc[20]))
df_train_x.shape
df_train_x.max().max()
df_train_x = df_train_x / 255
df_test = df_test / 255 
df_train_x.max().max()
clf = MLPClassifier(hidden_layer_sizes=(50, 10))
clf.fit(df_train_x, df_train_y)
train_result = clf.predict_proba(df_train_x)
def logloss(df_pred, df_correct):
    log_los = 0
    for i, index in enumerate(df_correct.values):
        log_los += -log(df_pred[i, index])
    log_los /= len(df_correct)
    return log_los
loglos_train = logloss(train_result, df_train_y)
print('Train loglos: {}'.format(loglos_train))
result = clf.predict(df_test)
plt.matshow(df_test.iloc[0].values.reshape(28,28))
print('Predict number: {}'.format(result[0]))
plt.matshow(df_test.iloc[3].values.reshape(28,28))
print('Predict number: {}'.format(result[3]))
df_train_y.plot.hist(grid=True,bins=10)
pd.DataFrame(result).plot.hist(grid=True,bins=10)