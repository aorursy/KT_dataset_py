import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



X_train = pd.read_json("../input/train.json")

# Any results you write to the current directory are saved as output.

Y_train = X_train['interest_level']

del X_train['interest_level']

X_train
print(list(X_train.columns))

s = set()

for i in range(X_train.shape[0]):

    s.add(len(X_train['photos'].iloc[i]))

print(s)