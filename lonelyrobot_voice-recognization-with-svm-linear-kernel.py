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
voice_dataframe = pd.read_csv('../input/voice.csv')
from sklearn import svm
model = svm.SVC(kernel='linear')
voice_dataframe['label'] = voice_dataframe['label'] == 'male'
voice_dataframe = voice_dataframe.sample(frac=1)

voice_dataframe_train = voice_dataframe.head(3000)

voice_dataframe_test = voice_dataframe.tail(100)
y_train = voice_dataframe_train['label'].data

X_train = voice_dataframe_train.drop('label', 1).values

y_test = voice_dataframe_test['label'].data

X_test = voice_dataframe_test.drop('label', 1).values
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)

print(sum(y_predicted == y_test))