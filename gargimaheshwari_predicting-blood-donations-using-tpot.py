import pandas as pd

from sklearn.model_selection import train_test_split

from tpot import TPOTClassifier

from sklearn.metrics import roc_auc_score



transfusion = pd.read_csv("../input/donations.csv")

transfusion.head()
transfusion.rename(

    columns={'V1':'Recency (months)',

             'V2':'Frequency(times)',

             'V3':'Monetary (c.c. blood)',

             'V4':'Time (months)',

             'Class':'Target'},

    inplace=True

)



transfusion.head()
transfusion.info()
transfusion['Target'] = transfusion['Target'].replace([1, 2], [0, 1])



display(transfusion['Target'].value_counts(normalize = True))
X_train, X_test, y_train, y_test = train_test_split(

    transfusion.drop(columns = 'Target'),

    transfusion.Target,

    test_size = 0.25,

    random_state = 42,

    stratify = transfusion.Target

)
tpot = TPOTClassifier(

    generations = 5,

    population_size = 20,

    verbosity = 2,

    scoring = 'roc_auc',

    random_state = 42,

    disable_update_check = True,

    config_dict = 'TPOT light'

)

tpot.fit(X_train, y_train)



tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])

print(f'\nAUC score: {tpot_auc_score:.4f}')



print('\nBest pipeline steps:', end='\n')

for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):

    print(f'{idx}. {transform}')
display(X_train.var())
import numpy as np



X_train_normed, X_test_normed = X_train.copy(), X_test.copy()

col_to_normalize = "Monetary (c.c. blood)"



# Log normalization

for df_ in [X_train_normed, X_test_normed]:

    df_['Monetary_log'] = np.log(df_[col_to_normalize])

    df_.drop(columns = col_to_normalize, inplace=True)



display(X_train_normed.var())
from sklearn import linear_model



logreg = linear_model.LogisticRegression(

    solver='liblinear',

    random_state=42

)



logreg.fit(X_train_normed, y_train)



logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test_normed)[:, 1])

print(f'\nAUC score: {logreg_auc_score:.4f}')