import time

start_time = time.time()
!pip install pycaret
import pandas as pd

import numpy as np

from pycaret.classification import *
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

display(train.head())

display(test.head())
modelo = setup(data=train, target='Survived', 

                #Pycaret tem a possibilidade de excluir colunas no modelo

                ignore_features = ['Name','Cabin','PassengerId', 'Ticket'], session_id=123,

                silent=True)
compare_models()
cat = create_model('catboost')
gbc = create_model('gbc')
tuned_cat = tune_model('catboost')
tuned_gbc = tune_model('gbc')
## Depois de finalizar o tunning e a criação do modelo salvamos o modelo finalizado.

final_gbc = finalize_model(gbc)
plot_model(estimator=final_gbc, plot='confusion_matrix')
plot_model(estimator=final_gbc, plot='auc')
plot_model(estimator=final_gbc, plot='feature')
interpret_model(cat)
ids = test['PassengerId']

test.drop(['PassengerId', 'Name','Ticket', 'Cabin'], axis=1, inplace=True)
pred = predict_model(final_gbc, data=test)
a = pd.DataFrame({'PassengerId': ids,

                 'Survived': pred['Label']})

a.to_csv('primeiro_modelo.csv', index=False)
print(f"This kernel took {(time.time() - start_time)/60:.2f} minutes to run")