import numpy as np

import pandas as pd

from patsy import dmatrices, build_design_matrices



train = pd.read_csv('../input/train_ver2.csv', nrows=5000).set_index('ncodpers')

test = pd.read_csv('../input/test_ver2.csv', nrows=None).set_index('ncodpers')



ycols = np.array(['ind_ahor_fin_ult1',

		 'ind_aval_fin_ult1',

		 'ind_cco_fin_ult1',

		 'ind_cder_fin_ult1',

		 'ind_cno_fin_ult1',

		 'ind_ctju_fin_ult1',

		 'ind_ctma_fin_ult1',

		 'ind_ctop_fin_ult1',

		 'ind_ctpp_fin_ult1',

		 'ind_deco_fin_ult1',

		 'ind_deme_fin_ult1',

		 'ind_dela_fin_ult1',

		 'ind_ecue_fin_ult1',

		 'ind_fond_fin_ult1',

		 'ind_hip_fin_ult1',

		 'ind_plan_fin_ult1',

		 'ind_pres_fin_ult1',

		 'ind_reca_fin_ult1',

		 'ind_tjcr_fin_ult1',

		 'ind_valo_fin_ult1',

		 'ind_viv_fin_ult1',

		 'ind_nomina_ult1',

		 'ind_nom_pens_ult1',

		 'ind_recibo_ult1'])

formula = ' + '.join(ycols) + """ ~ indrel """

y, X = dmatrices(formula, data=train, NA_action='drop', return_type='dataframe')

X_test = build_design_matrices([X.design_info], data=test, return_type='dataframe')[0]
from sklearn.pipeline import Pipeline

import sklearn.linear_model as linmod

from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import cross_val_score



model = Pipeline([('scaler', StandardScaler()),

                  ('ridge', linmod.Ridge())])



run_cv = True

if run_cv:

    scores = cross_val_score(model, X, y, cv=5, n_jobs=4, scoring='neg_mean_squared_error')

    print('{:.3f} +/- {:.3f}'.format(np.mean(scores), np.std(scores)))



model.fit(X, y)

preds = [[p > 0.5 for p in cust] for cust in model.predict(X_test)]

preds = [' '.join([ycols[p == True]]) for p in preds]

preds = pd.DataFrame({'added_products': preds}, index=X_test.index).to_csv('submission.csv')