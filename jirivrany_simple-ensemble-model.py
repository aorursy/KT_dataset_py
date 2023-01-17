import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

files = ['../input/original-submission-sample/sample_submission.csv',

         '../input/example-submission-file/submission.csv']
#one hot encoded result

y_final = np.zeros((28000, 10))



for fname in files:

    df1 = pd.read_csv(fname)

    labels = np.array(df1.pop('Label'))

    labels = LabelEncoder().fit_transform(labels)[:, None]

    labels = OneHotEncoder().fit_transform(labels).todense()

    y_final += labels



print(y_final)
predictions = np.argmax(y_final, axis=1)

print(predictions)
submission = pd.DataFrame(data={'ImageId': (np.arange(len(predictions)) + 1), 'Label': predictions})

submission.to_csv('submission-ensembled-model.csv', index=False)

submission.tail()   
