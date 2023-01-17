import re

import pandas as pd

from sklearn.preprocessing import LabelEncoder





def get_relation(x):

    if x == "Other":

        return x

    else:

        return x[:-7]





def get_direction(x):

    if x == "Other":

        return 0

    else:

        return int(x[-5])





class DataReader:

    """

    DataReader for train and test set.

    Also encodes and decodes the target.

    Parameters

    ----------

    nominal_regex: string

        Regex describing nominals in the data

    """

    def __init__(self, nominal_regex=r"</?e\d>.+?</e\d>"):

        self.relation_le = LabelEncoder()

        self.re_nominal = re.compile(nominal_regex)



    def _get_nominals(self, text):

        nominals_orig = self.re_nominal.findall(text)



        nominals = list(nominals_orig)

        for i in range(2):

            nominals[i] = nominals_orig[i][4:-5]

            text = text.replace(nominals_orig[i], " " + nominals[i] + " ")



        return text, nominals



    def _read_text_file(self, filename, step_size):

        with open(filename) as f:

            all_lines = f.readlines()



        data = list()



        for text in all_lines[0::step_size]:

            text = text.split("\t")

            row_index, text = text[0], text[1]

            text = text[1:-2]  # remove " and new line



            text, nominals = self._get_nominals(text)



            data.append({"row_index": row_index, "e1": nominals[0], "e2": nominals[1], "text": text})



        return pd.DataFrame(data), all_lines



    def read_train_file(self, filename):

        """Reads train data from txt.

        Parameters

        ----------

        filename : string

        Returns

        -------

        df : pd.DataFrame

            dataframe including index, text, nominals, target and sample weights

        """

        df, all_lines = self._read_text_file(filename, step_size=4)

        df["target"] = [label[:-1] for label in all_lines[1::4]]  # remove new line



        df["relation"] = df["target"].apply(get_relation)

        df["direction"] = df["target"].apply(get_direction)



        # encode the target into 2 digit number (relation_id, direction)

        df["target_code"] = self.encode_target(df["relation"], df["direction"])



        return df



    def read_test_file(self, filename):

        """Reads test data from txt.

        Parameters

        ----------

        filename : string

        Returns

        -------

        df : pd.DataFrame

            dataframe including index, text and nominals

        """

        df, _ = self._read_text_file(filename, step_size=1)

        return df



    def encode_target(self, relation, direction):

        """Encodes the target into 2 digit code

        Parameters

        ----------

        relation : pd.Series

            Relation between the nominals

        direction: pd.Series

            Direction of the relation

        Returns

        -------

        encoded_target : pd.Series

        """

        self.relation_le.fit(relation)

        encoded_target = 10 * self.relation_le.transform(relation) + direction

        return encoded_target



    def decode_target(self, encoded_target):

        """Decodes the target code into its original string

        Parameters

        ----------

        encoded_target : pd.Series

        Returns

        -------

        decoded_target : pd.Series

        """

        directions = ["", "(e1,e2)", "(e2,e1)"]

        decoded_target = self.relation_le.inverse_transform(encoded_target // 10)

        decoded_target += (encoded_target % 10).apply(lambda x: directions[x])

        return decoded_target

data_reader = DataReader()

df = data_reader.read_train_file("../input/ing-nlp-event-2020/TRAIN_FILE.TXT")

df.head()
df["target"].value_counts()
test_df = data_reader.read_test_file("../input/ing-nlp-event-2020/TEST_FILE.txt")

test_df.head()
import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



kfold = StratifiedKFold(4, shuffle=True, random_state=0)





le = LabelEncoder()

y = le.fit_transform(df["target_code"])



y_oof = np.zeros(y.shape)



for fold_id, (train_ind, val_ind) in enumerate(kfold.split(df["target_code"], df["target_code"])):

    train_df, val_df = df.iloc[train_ind].copy(), df.iloc[val_ind].copy()

    tfidf = TfidfVectorizer(min_df=10)

    X_train = tfidf.fit_transform(train_df["text"])

    X_val = tfidf.transform(val_df["text"])

    

    lr = LogisticRegression(random_state=1000)

    lr.fit(X_train, y[train_ind])



    y_oof[val_ind] = lr.predict(X_val)



print("CV model performance = {}".format(accuracy_score(y, y_oof)))
lr_final = LogisticRegression()

lr_final.fit(tfidf.fit_transform(df["text"]), y)

y_test = lr_final.predict(tfidf.transform(test_df["text"]))

test_df["Category"] = data_reader.decode_target(pd.Series(le.inverse_transform(y_test)))

test_df.head()



test_df.rename(columns = {"row_index": "Id"}).to_csv("basic_model.csv", index=False, columns=["Id", "Category"])