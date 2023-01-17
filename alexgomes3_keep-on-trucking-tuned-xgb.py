import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import imputation

from matplotlib import pyplot as plt



from sklearn.metrics import fbeta_score, make_scorer, precision_score, recall_score

from sklearn.model_selection import cross_val_score



import seaborn as sns

from sklearn.metrics import precision_score, recall_score

from sklearn.preprocessing import StandardScaler, FunctionTransformer

from sklearn.pipeline import make_pipeline



import xgboost as xgb
def load_and_clean_df(filename):

    _df = pd.read_csv(filename, na_values='na')

    _df = _df.set_index("ID")

    _df = _df.rename({'class':'target'}, axis=1).astype(np.number)

    return _df



def normalize_hist(df, var_group):

    _df = df.copy()

    l_cols_int = [c for c in df.columns if c[:2] == var_group]

    _df[l_cols_int] = _df[l_cols_int].div(_df[l_cols_int].sum(axis=1), axis=0)

    return _df

    

def normalize_histograms_in_df(df):

    l_hist_var_groups = [c[:2] for c in df.columns if c[-1]=="1"]

    output = df.copy()

    for hist_var_group in l_hist_var_groups:

        output = normalize_hist(output, hist_var_group)

    return output



def split_x_y(df, target_col = 'target'):

    return df.drop(target_col,axis=1), df[target_col]
df = load_and_clean_df("../input/training_data_set.csv")

df_test = load_and_clean_df("../input/test_data_set.csv")



df.head()
normalizer = FunctionTransformer(normalize_histograms_in_df, validate=False) 

imputer = imputation.Imputer() 

standardizer = StandardScaler()

data_prep_pipeline = make_pipeline(normalizer, imputer, standardizer)



X_train, X_val, y_train, y_val = train_test_split(*split_x_y(df), test_size=0.2, random_state=42)
%%time

pos_weight = (df.target==0).sum()/df.target.sum()



m2 = xgb.XGBClassifier(objective='binary:logistic',

                       max_depth=3,

                        learning_rate=0.1,

                        base_score =0.95,

                        gamma=0.3,

                        reg_alpha=0.3,

                        subsample=0.9,

                        colsample_bytree=0.9,

                        n_estimators=500,

                        scale_pos_weight = pos_weight

                        ,n_jobs=4

                        ,gpu_id = 0

                        ,max_bin = 16

                        ,tree_method = 'gpu_hist'

                      )



pipe2 = make_pipeline(normalizer, imputer, standardizer, m2)

pipe2.fit(X_train, y_train)
y_pred_val = pipe2.predict_proba(X_val)[:,1]



lx = np.linspace(0.0001,.3,100)

f_p = np.vectorize(lambda thr: precision_score(y_val, y_pred_val>thr))

f_r = np.vectorize(lambda thr: recall_score(y_val, y_pred_val>thr))

f_b = np.vectorize(lambda thr: fbeta_score(y_val, y_pred_val>thr, 7.07))

sns.lineplot(lx,f_p(lx), label = "Precision")

sns.lineplot(lx,f_r(lx), label = "Recall")

sns.lineplot(lx,f_b(lx), label = r"$f_\beta$")

plt.xlabel('Decision threshold')

plt.title(fr"$max\,f_\beta={f_b(lx).max():3.3} \,for\, thr={lx[f_b(lx).argmax()]:3.3}$")

plt.legend()

plt.show()
XGB_model_optimal_params = xgb.XGBClassifier(objective='binary:logistic',

                       max_depth=3,

                        learning_rate=0.1,

                        base_score =0.95,

                        gamma=0.3,

                        reg_alpha=0.3,

                        subsample=0.9,

                        colsample_bytree=0.9,

                        n_estimators=500,

                        scale_pos_weight = pos_weight

                        ,n_jobs=4

                        ,gpu_id = 0

                        ,max_bin = 16

                        ,tree_method = 'gpu_hist')

pipe2 = make_pipeline(normalizer, imputer, standardizer, XGB_model_optimal_params)

pipe2.fit(*split_x_y(df))
y_pred = pipe2.predict_proba(df_test)[:,1]>0.07



df_predictions = pd.DataFrame(y_pred.astype(int), index = df_test.index, columns = ["Predicted"])

df_predictions.to_csv('XGB_submission_balanced.csv')