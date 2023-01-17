import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("../input/TRAIN.csv")

test = pd.read_csv("../input/TEST.csv", index_col=47)



test.head()
# sns.pairplot(data=data, hue="readmitted_NO", dropna=True)
medication_features = ['metformin',

       'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',

       'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',

       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',

       'tolazamide', 'examide', 'citoglipton', 'insulin',

       'glyburide-metformin', 'glipizide-metformin',

       'glimepiride-pioglitazone', 'metformin-rosiglitazone',

       'metformin-pioglitazone']



cat_col = ['race', 'gender', 'age', 'weight', 'payer_code', 'medical_specialty', 'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed']

num_cols = ['time_in_hospital', 'num_lab_procedures',

       'num_procedures', 'num_medications', 'number_outpatient',

       'number_emergency', 'number_inpatient',

       'number_diagnoses']

target_col = ['readmitted_NO']
from copy import deepcopy

def process_data(df, labelencoder=dict(), cat_cols = [], encode=True):

    df = deepcopy(df)

    df.replace("?", 'NaN', inplace=True)

    df.replace(np.nan, 'NaN', inplace=True)

    df.medical_specialty = df.medical_specialty.str.replace("&",'and')

    df.medical_specialty = df.medical_specialty.str.replace("/",'or')

    df.diag_1 = df.diag_1.str.extract(r'(\d+).')[0]

    df.diag_2 = df.diag_2.str.extract(r'(\d+).')[0]

    df.diag_3 = df.diag_3.str.extract(r'(\d+).')[0]

    

    def process_split_medical_specialty(dd):

        col = 'medical_specialty'

        arr = list()

        for i in dd.medical_specialty.str.split('-'):

            if type(i).__name__ == 'list':

                if len(i) == 1:

                    i = ["NaN"] + i

                if len(i) > 2:

                    i = [i[0]] + ['and'.join(i[1:])]

            elif i == "NaN":

                i = ["NaN", "NaN"]

            arr.append(i)

        

#         d = pd.DataFrame(arr, index=dd.index, columns=[0,1])

        arr = np.array(arr)

        dd['medical_field'] = arr[:, 0]

        dd['medical_specialty'] = arr[:, 1]

        return dd



    df.max_glu_serum = df.max_glu_serum.replace("None", "NaN")

    df.A1Cresult = df.A1Cresult.replace("None", "NaN")

    df.age = df.age.str.extract(r'\[(\d+)-')[0]

    df.weight = df.weight.str.extract(r'\[(\d+)-')[0]

    df = process_split_medical_specialty(df)

    cat_cols += ['medical_field', "diag_1", 'diag_2', 'diag_3', 'weight', 'age']

    df.replace(np.nan, 'NaN', inplace=True)



#     for col in df.columns:

#         try:

#             print(col , df[col][df[col] != "NaN"].iloc[0])

#         except:

#             print("%s has all values null", col)



    if encode == True:

        for col in cat_cols:

            labelencoder[col] = LabelEncoder().fit(df[col])

            df[col] = labelencoder[col].transform(df[col])

    else:

        for col in cat_cols:

            try:

                df[col] = labelencoder[col].transform(df[col])

            except:

                print(col)

                df[col] = df[col].astype(str)

    df = df.astype(float)

    return df, labelencoder
X, y = data.drop(columns=["readmitted_NO"]), data.readmitted_NO

d = pd.concat([X, test], axis=0, sort=False, keys=['X', 'test'])

processed_Xandtest, le = process_data(d, cat_cols=cat_col+medication_features, encode=True)

processed_X = processed_Xandtest.loc['X']

processed_test = processed_Xandtest.loc['test']

best_cols = ['num_lab_procedures', 'num_medications', 'number_diagnoses']

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(processed_X[best_cols], y, test_size=0.25, random_state=2019)
# from sklearn.decomposition import PCA

# pca = PCA(n_components = 5)

# X_train_pca = pca.fit_transform(X_train)

# X_test_pca = pca.transform(X_test)



# # processed_X_pca = pca.fit_transform(processed_X)

# # processed_test_pca = pca.transform(processed_test)
processed_X = processed_X[best_cols]

processed_test = processed_test[best_cols]
from sklearn.cluster import KMeans

from sklearn.mixture import GaussianMixture

from sklearn.cluster import DBSCAN

from sklearn.cluster import SpectralClustering

# import hdbscan



# clf = SpectralClustering()

# clf = DBSCAN(eps=2.5, min_samples=6)

clf = KMeans(n_clusters=2, algorithm='elkan')

# clf = GaussianMixture(n_components=2)



clf.fit(processed_X, y)

# pred = clf.fit_predict(X_train, y_train)

# clf.fit(X_train, y_train)



pred = clf.predict(processed_test)

# pred = clf.predict(X_test)
# d = pd.DataFrame(columns=['pred', 'true_label'])

# d.pred = pred

# d.true_label = y_test

# # d.true_label = y_train



# d.groupby(['true_label', 'pred']).size()
# from sklearn.metrics import accuracy_score

# accuracy_score(y_true=y_test, y_pred=pred)

# # accuracy_score(y_true=y_train, y_pred=pred)
pred.shape, test.shape
test['target'] = pred

test[['target']].to_csv("kmeans.csv", index=True)