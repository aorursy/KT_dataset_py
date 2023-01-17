!pip install pycaret
import pandas as pd
from pycaret.classification import *
data = pd.read_csv("../input/titanic/train.csv")
data.head()
Setup = setup(data= data, target='Survived',remove_outliers=True,normalize=True)
compare_models()
dt = create_model('dt')
test = pd.read_csv("../input/titanic/test.csv")
pred = predict_model(dt, data = test)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': pred.Label})
output.to_csv('submission.csv', index=False)
output.head()