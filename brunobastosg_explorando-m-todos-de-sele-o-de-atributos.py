import pandas as pd



df = pd.read_csv("../input/sonar.csv")



df.head()
X = df.iloc[:,:-1]

y = df.iloc[:, -1]



X.columns.size
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



selector_chi2 = SelectKBest(chi2, k=8)

selector_chi2.fit_transform(X, y)



cols_chi2 = selector_chi2.get_support(indices=True)

df_chi2 = df.iloc[:,cols_chi2]



list(df_chi2.columns)
from sklearn.feature_selection import f_classif



selector_f_classif = SelectKBest(f_classif, k=8)

selector_f_classif.fit_transform(X, y)



cols_f_classif = selector_f_classif.get_support(indices=True)

df_f_classif = df.iloc[:,cols_f_classif]



list(df_f_classif.columns)
from sklearn.feature_selection import mutual_info_classif



selector_mutual_info_classif = SelectKBest(mutual_info_classif, k=8)

selector_mutual_info_classif.fit_transform(X, y)



cols_mutual_info_classif = selector_mutual_info_classif.get_support(indices=True)

df_mutual_info_classif = df.iloc[:,cols_mutual_info_classif]



list(df_mutual_info_classif.columns)
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
svc = SVC(kernel="linear", C=1)

rfe = RFE(estimator=svc, n_features_to_select=8, step=1)

rfe.fit(X, y)



cols_svc = rfe.support_



df_svc = X.iloc[:,cols_svc]



list(df_svc.columns)