import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')
data.info()
X = data.drop('label', axis=1)
Y = data.label
model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
model.fit(X, Y)
print(model.score(X, Y))
test_data = pd.read_csv('../input/test.csv')
results = model.predict(test_data)
sub = pd.DataFrame({'ImageId':test_data.index +1,
                    'Label':results})
sub[['ImageId','Label']].to_csv('submission.csv', index=False)