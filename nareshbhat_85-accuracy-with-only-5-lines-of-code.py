%%capture
!pip install pycaret
import pandas as pd
from pycaret.classification import *
train = pd.read_csv('../input/titanic/train.csv')
train.head()
test = pd.read_csv('../input/titanic/test.csv')
test.head()
classification_setup = setup(data= train, target='Survived',remove_outliers=True,normalize=True,session_id=8351,
                            ignore_features= ['Name'])
compare_models()
ridge = create_model('ridge')
pred = predict_model(ridge, data = test)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': pred.Label})
output.to_csv('submission.csv', index=False)
output.head()
