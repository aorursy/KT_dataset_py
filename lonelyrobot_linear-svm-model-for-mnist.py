import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import svm



training_data = pd.read_csv('../input/train.csv')
training_data = training_data.sample(frac=1)



y_train = training_data['label'].head(40000).values

y_test = training_data['label'].tail(2000).values



training_data = training_data.drop('label', 1)



X_train = training_data.head(40000).values

X_test = training_data.tail(2000).values
# Our model is the polynomial kernel

model = svm.SVC(kernel='poly', degree=2)

model.fit(X_train, y_train)



print(model.score(X_test, y_test))
test_data = pd.read_csv('../input/test.csv')



X_test = test_data.values



# Getting the prediciton.

y_pred_test = model.predict(X_test)
result_df = pd.DataFrame(data=y_pred_test, index=np.arange(1, 28001))



result_df.index.name = 'ImageId'

result_df.columns = ['Label']
result_df.to_csv(path_or_buf='output.csv')