import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from json import JSONDecoder, JSONDecodeError  # for reading the JSON data files

import re  # for regular expressions

import os  # for os related operations
print(os.listdir("../input"))
def decode_obj(line, pos=0, decoder=JSONDecoder()):

    no_white_space_regex = re.compile(r'[^\s]')

    while True:

        match = no_white_space_regex.search(line, pos)

        if not match:

            return

        pos = match.start()

        try:

            obj, pos = decoder.raw_decode(line, pos)

        except JSONDecodeError as err:

            print('Oops! something went wrong. Error: {}'.format(err))

        yield obj
def get_obj_with_last_n_val(line, n):

    obj = next(decode_obj(line))  # type:dict

    id = obj['id']

    class_label = obj['classNum']



    data = pd.DataFrame.from_dict(obj['values'])  # type:pd.DataFrame

    data.set_index(data.index.astype(int), inplace=True)

    last_n_indices = np.arange(0, 60)[-n:]

    data = data.loc[last_n_indices]



    return {'id': id, 'classType': class_label, 'values': data}
def convert_json_data_to_csv(data_dir: str, file_name: str):

    """

    Generates a dataframe by concatenating the last values of each

    multi-variate time series. This method is designed as an example

    to show how a json object can be converted into a csv file.

    :param data_dir: the path to the data directory.

    :param file_name: name of the file to be read, with the extension.

    :return: the generated dataframe.

    """

    fname = os.path.join(data_dir, file_name)



    all_df, labels, ids = [], [], []

    with open(fname, 'r') as infile: # Open the file for reading

        for line in infile:  # Each 'line' is one MVTS with its single label (0 or 1).

            obj = get_obj_with_last_n_val(line, 1)

            all_df.append(obj['values'])

            labels.append(obj['classType'])

            ids.append(obj['id'])



    df = pd.concat(all_df).reset_index(drop=True)

    df = df.assign(LABEL=pd.Series(labels))

    df = df.assign(ID=pd.Series(ids))

    df.set_index([pd.Index(ids)])

    # Uncomment if you want to save this as CSV

    # df.to_csv(file_name + '_last_vals.csv', index=False)

    return df
path_to_data = "../input"

file_name = "fold3Training.json"



df = convert_json_data_to_csv(path_to_data, file_name)  # shape: 27006 X 27

print('df.shape = {}'.format(df.shape))

# print(list(df))
df = df.dropna()  # shape: 26666 X 27

print('df.shape = {}'.format(df.shape))
t = (2/3) * df.shape[0]

df_train = df[df['ID'] <= t]  # shape: 18004 X 27

df_val = df[df['ID'] > t]  # shape: 9002 X 27

print('df_train.shape = {}'.format(df_train.shape))

print('df_val.shape = {}'.format(df_val.shape))
from sklearn import svm

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score
# Separate values and labels columns

df_train_data = df_train.iloc[:, :-2]  # all columns excluding 'ID' and 'LABEL'

df_train_labels = pd.DataFrame(df_train.LABEL)  # only 'LABEL' column



df_val_data = df_val.iloc[:, :-2]  # all columns excluding 'ID' and 'LABEL'

df_val_labels = pd.DataFrame(df_val.LABEL)  # only 'LABEL' column



# Train a simple SVM as an example

svm_c = 1000

svm_gamma = 0.01

clf = svm.SVC(gamma=svm_gamma, C=svm_c, max_iter=-1, verbose=1, shrinking=True, random_state=42)

clf.fit(df_train_data, np.ravel(df_train_labels))
# Test the model against the validation set

pred_labels = clf.predict(df_val_data)



# Evaluate the predictions

scores = confusion_matrix(df_val_labels, pred_labels).ravel()

tn, fp, fn, tp = scores

print('TN:{}\tFP:{}\tFN:{}\tTP:{}'.format(tn, fp, fn, tp))

f1 = f1_score(df_val_labels, pred_labels, average='binary', labels=[0, 1])

print('f1-score = {}'.format(f1))