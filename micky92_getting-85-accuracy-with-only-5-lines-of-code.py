!pip install pycaret
import pandas as pd

from pycaret.classification import create_model,setup

from pycaret import classification
train = pd.read_csv('../input/titanic/train.csv')

train.head()
test = pd.read_csv('../input/titanic/test.csv')

test.head()
classification_setup = setup(data= train, target='Survived',remove_outliers=True,normalize=True,normalize_method='minmax',

                            ignore_features= ['Name'])
classification.compare_models()
rc = create_model('ridge')
pred = classification.predict_model(rc, data = test)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': pred.Label})

output.to_csv('submission.csv', index=False)
output.head()