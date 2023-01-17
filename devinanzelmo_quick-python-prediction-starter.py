import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn import ensemble, model_selection, metrics



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
credit_card = pd.read_csv('../input/creditcard.csv')
credit_card.head()
labels = credit_card['Class']

times = credit_card['Time']

credit_card.drop(['Class','Time'],axis=1,inplace=True)
labels.value_counts()
strat_kfold = model_selection.StratifiedKFold(n_splits=5,shuffle=True, random_state=11111)
rf = ensemble.RandomForestClassifier(n_jobs=-1)

scores = model_selection.cross_val_score(rf, credit_card, labels, cv=strat_kfold,scoring='roc_auc')

scores