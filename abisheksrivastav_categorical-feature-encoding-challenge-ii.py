# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#pd = df = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")
import  pandas as pd

from sklearn import preprocessing



class CategoricalFeatures:

    def __init__(self, df, categorical_features, encoding_type, handle_na=False):

        """

        df: pandas dataframe

        categorical_features: list of column names, e.g. ["ord_1", "nom_0"......]

        encoding_type: label, binary, ohe

        handle_na: True/False

        """

        self.df = df

        self.cat_feats = categorical_features

        self.enc_type = encoding_type

        self.handle_na = handle_na

        self.label_encoders = dict()

        self.binary_encoders = dict()

        self.ohe = None



        if self.handle_na:

            for c in self.cat_feats:

                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-9999999")

        self.output_df = self.df.copy(deep=True)

    

    def _label_encoding(self):

        for c in self.cat_feats:

            lbl = preprocessing.LabelEncoder()

            lbl.fit(self.df[c].values)

            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)

            self.label_encoders[c] = lbl

        return self.output_df

    

    def _label_binarization(self):

        for c in self.cat_feats:

            lbl = preprocessing.LabelBinarizer()

            lbl.fit(self.df[c].values)

            val = lbl.transform(self.df[c].values)

            self.output_df = self.output_df.drop(c, axis=1)

            for j in range(val.shape[1]):

                new_col_name = c + f"__bin_{j}"

                self.output_df[new_col_name] = val[:, j]

            self.binary_encoders[c] = lbl

        return self.output_df



    def _one_hot(self):

        ohe = preprocessing.OneHotEncoder()

        ohe.fit(self.df[self.cat_feats].values)

        return ohe.transform(self.df[self.cat_feats].values)



    def fit_transform(self):

        if self.enc_type == "label":

            return self._label_encoding()

        elif self.enc_type == "binary":

            return self._label_binarization()

        elif self.enc_type == "ohe":

            return self._one_hot()

        else:

            raise Exception("Encoding type not understood")

    

    def transform(self, dataframe):

        if self.handle_na:

            for c in self.cat_feats:

                dataframe.loc[:, c] = dataframe.loc[:, c].astype(str).fillna("-9999999")



        if self.enc_type == "label":

            for c, lbl in self.label_encoders.items():

                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)

            return dataframe



        elif self.enc_type == "binary":

            for c, lbl in self.binary_encoders.items():

                val = lbl.transform(dataframe[c].values)

                dataframe = dataframe.drop(c, axis=1)

                

                for j in range(val.shape[1]):

                    new_col_name = c + f"__bin_{j}"

                    dataframe[new_col_name] = val[:, j]

            return dataframe



        elif self.enc_type == "ohe":

            return self.ohe(dataframe[self.cat_feats].values)

        

        else:

            raise Exception("Encoding type not understood")

                



if __name__ == "__main__":

    import pandas as pd

    from sklearn import linear_model

    df = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")

    

    df_test = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")

    sample = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/sample_submission.csv")



    train_len = len(df)



    df_test["target"] = -1

    full_data = pd.concat([df, df_test])



    cols = [c for c in df.columns if c not in ["id", "target"]]

    cat_feats = CategoricalFeatures(full_data, 

                                    categorical_features=cols, 

                                    encoding_type="ohe",

                                    handle_na=True)

    full_data_transformed = cat_feats.fit_transform()

    

    X = full_data_transformed[:train_len, :]

    X_test = full_data_transformed[train_len:, :]



    clf = linear_model.LogisticRegression()

    clf.fit(X, df.target.values)

    preds = clf.predict_proba(X_test)[:, 1]

    

    submission = pd.DataFrame({'id': df_test.id, 'target': preds})

    submission.to_csv('submission.csv', index=False)

    

    #sample.loc[:, "target"] = preds

   # sample.to_csv("submission.csv", index=False)