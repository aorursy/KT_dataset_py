import numpy as np
import csv
import scipy
from scipy.io import wavfile
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
def custom_database_import(in_path):
    index_list = os.listdir(in_path)
    in_all_audios = []
    # in_class_cnt = {}
    in_y = []

    index_list = sorted(index_list, key=lambda x: int(x[:-6]))
    
    for elem in index_list:
        in_y.append(elem[-5])

    for filename in index_list:
        filename = in_path + f"{filename}"
        in_all_audios.append(scipy.io.wavfile.read(filename, mmap=False))

    out_y = np.array(in_y)
    return in_all_audios, out_y


def custom_eval_database_import(in_path):
    index_list = os.listdir(in_path)
    in_all_audios = []

    index_list = sorted(index_list, key=lambda x: int(x[:-4]))

    for filename in index_list:
        filename = in_path + f"{filename}"
        in_all_audios.append(scipy.io.wavfile.read(filename, mmap=False))

    return in_all_audios



def custom_csv_print(in_labels, filename):
    list_to_print = []
    for index in range(0, len(in_labels)):
        row_to_print = []
        row_to_print.append(index)
        row_to_print.append(in_labels[index])
        list_to_print.append(row_to_print)

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Predicted'])
        for index in range(0, len(list_to_print)):
            writer.writerow(list_to_print[index])
    return


def custom_preprocess(in_all_audios):
    frequency_preprocessed = []
    all_normalized_audios = []
    all_samples_processed = []

    # Normalization
    for i in range(0, len(in_all_audios)):
        single_normalized_audio = in_all_audios[i][1] / np.max(np.abs(in_all_audios[i][1]))
        all_normalized_audios.append(single_normalized_audio)

    # Frequency Domain
    for i in range(0, len(all_normalized_audios)):
        freq = np.abs(np.fft.fft(all_normalized_audios[i]))
        frequency_preprocessed.append(freq[:freq.shape[0]//2])

    # Sampling
    in_flag = 32
    for i in range(0, len(frequency_preprocessed)):
        single_sample_processed = []
        if in_flag == 32:
            single_sample_processed.append(
                np.mean(frequency_preprocessed[i][:1 * len(frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   1 * len(frequency_preprocessed[i]) // 32:2 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   2 * len(frequency_preprocessed[i]) // 32:3 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   3 * len(frequency_preprocessed[i]) // 32:4 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   4 * len(frequency_preprocessed[i]) // 32:5 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   5 * len(frequency_preprocessed[i]) // 32:6 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   6 * len(frequency_preprocessed[i]) // 32:7 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   7 * len(frequency_preprocessed[i]) // 32:8 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   8 * len(frequency_preprocessed[i]) // 32:9 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   9 * len(frequency_preprocessed[i]) // 32:10 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   10 * len(frequency_preprocessed[i]) // 32:11 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   11 * len(frequency_preprocessed[i]) // 32:12 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   12 * len(frequency_preprocessed[i]) // 32:13 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   13 * len(frequency_preprocessed[i]) // 32:14 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   14 * len(frequency_preprocessed[i]) // 32:15 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   15 * len(frequency_preprocessed[i]) // 32:16 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   16 * len(frequency_preprocessed[i]) // 32:17 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   17 * len(frequency_preprocessed[i]) // 32:18 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   18 * len(frequency_preprocessed[i]) // 32:19 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   19 * len(frequency_preprocessed[i]) // 32:20 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   20 * len(frequency_preprocessed[i]) // 32:21 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   21 * len(frequency_preprocessed[i]) // 32:22 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   22 * len(frequency_preprocessed[i]) // 32:23 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   23 * len(frequency_preprocessed[i]) // 32:24 * len(
                                                       frequency_preprocessed[i]) // 16]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   24 * len(frequency_preprocessed[i]) // 32:25 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   25 * len(frequency_preprocessed[i]) // 32:26 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   26 * len(frequency_preprocessed[i]) // 32:27 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   27 * len(frequency_preprocessed[i]) // 32:28 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   28 * len(frequency_preprocessed[i]) // 32:29 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   29 * len(frequency_preprocessed[i]) // 32:30 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   30 * len(frequency_preprocessed[i]) // 32:31 * len(
                                                       frequency_preprocessed[i]) // 32]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][
                                                   31 * len(frequency_preprocessed[i]) // 32:32 * len(
                                                       frequency_preprocessed[i]) // 32]))

        all_samples_processed.append(single_sample_processed)

    return all_samples_processed
all_test_audios, y = custom_database_import("../input/dev/")

X = np.array(custom_preprocess(all_test_audios))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

hyp_parameters = {
    "random_state": [0],
    "n_estimators": [100, 1000],
    "max_depth": [None, 2, 4],
    "max_features": ['auto', 'sqrt']
}

config_cnt = 0
tot_config = 2 * 3 * 2
max_f1 = 0

for config in ParameterGrid(hyp_parameters):
    config_cnt += 1
    print(f'Analizing config {config_cnt} of {tot_config} || Config: {config}')

    clf = RandomForestClassifier(**config)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    acc = accuracy_score(y_test_pred, y_test)
    p1, r1, f11, s1 = precision_recall_fscore_support(y_test, y_test_pred)
    macro_f1 = f11.mean()

    if macro_f1 > max_f1:
        max_f1 = macro_f1
        print(f"-----> Score: {macro_f1}")
        print()
hyp_parameters = {
    "random_state": [0],
    "hidden_layer_sizes": [100, 1000],
    "activation": ['logistic', 'relu'],
    "alpha": [0.00001, 0.0001, 0.001]
}

config_cnt = 0
tot_config = 2 * 2 * 3
max_f1 = 0

for config in ParameterGrid(hyp_parameters):
    config_cnt += 1
    print(f'Analizing config {config_cnt} of {tot_config} || Config: {config}')

    clf = MLPClassifier(**config)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    acc = accuracy_score(y_test_pred, y_test)
    p1, r1, f11, s1 = precision_recall_fscore_support(y_test, y_test_pred)
    macro_f1 = f11.mean()

    if macro_f1 > max_f1:
        max_f1 = macro_f1
        print(f"-----> Score: {macro_f1}")
        print()
all_eval_audios = custom_eval_database_import("../input/eval/")
X_eval = np.array(custom_preprocess(all_eval_audios))

forest_clf = RandomForestClassifier(max_depth=None, n_estimators=1000)
forest_clf.fit(X, y)
forest_y_final_pred = forest_clf.predict(X_eval)

MLP_clf = MLPClassifier(activation='relu', alpha=0.001, hidden_layer_sizes=1000)
MLP_clf.fit(X, y)
MLP_y_final_pred = MLP_clf.predict(X_eval)

custom_csv_print(forest_y_final_pred, 'forest_out')
custom_csv_print(MLP_y_final_pred, 'MLP_out')