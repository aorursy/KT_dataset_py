# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
! mkdir /kaggle/processed_data
# 1. Process data

# -*- coding: utf-8 -*- #

"""Preprocess Meta Information"""



import sys

sys.path.append('/kaggle/input/configutil')

sys.path.append('/kaggle/input/competitionutils')



import os

import gc

from PIL import Image

from pathlib import PosixPath

from argparse import ArgumentParser



import pandas as pd

import utils

import config





def argparse():

    """Parse Comndline Args."""

    usage_msg = """

\n  python {0} [-tr] [-te]\n

""".format(__file__,)

    parser = ArgumentParser(

        prog="preprocess.py", usage=usage_msg)



    parser.add_argument('-tr', "--train", dest="train", action='store_true')

    parser.add_argument('-te', "--test", dest="test", action='store_true')

    argvs = parser.parse_args()

    return argvs





def convert_parquet_to_images(

    parquet_file_path: PosixPath, image_dir_path: PosixPath

) -> bool:

    """Convert a parquet file to image files and save them"""

    df = pd.read_parquet(parquet_file_path)

    image_ids = df.image_id.values

    arrs = df.iloc[:, 1:].values

    del df

    gc.collect()



    for i, image_id in enumerate(image_ids):

        Image.fromarray(

            arrs[i, :].reshape(config.ORIGINAL_IMAGE_HIGHT, config.ORIGINAL_IMAGE_WIDTH)

        ).save(image_dir_path / "{}.png".format(image_id))

    del arrs

    del image_ids

    gc.collect()





def preprocess_parquet_files(process_train: bool, process_test: bool) -> None:

    """Read parquet files and convert them into images."""

    if process_train:

        # # train

        print("preprocess train parquet files")

        if os.path.isdir(config.TRAIN_IMAGES_DIR):

            print("train images dir already exists!")

            pass

        else:

            os.mkdir(config.TRAIN_IMAGES_DIR)

            for i in range(config.PARUET_FILE_NUM):

                pqt_file_name = "train_image_data_{}.parquet".format(i)

                parquet_file_path = config.RAW_DATA / pqt_file_name

                with utils.timer("convert {} to png files.".format(pqt_file_name)):

                    convert_parquet_to_images(parquet_file_path, config.TRAIN_IMAGES_DIR)



    if process_test:

        # # test

        print("preprocess test parquet files")

        if os.path.isdir(config.TEST_IMAGES_DIR):

            print("test images dir already exists!")

            pass

        else:

            os.mkdir(config.TEST_IMAGES_DIR)

            for i in range(config.PARUET_FILE_NUM):

                pqt_file_name = "test_image_data_{}.parquet".format(i)

                parquet_file_path = config.RAW_DATA / pqt_file_name

                with utils.timer("convert {} to png files.".format(pqt_file_name)):

                    convert_parquet_to_images(parquet_file_path, config.TEST_IMAGES_DIR)





def preprocess_meta_info_files(process_train: bool, process_test: bool) -> None:

    """Preprocess Train and Test Meta Info."""

    if process_train:

        with utils.timer("preprocess train meta file"):

            train = pd.read_csv(config.RAW_DATA / "train.csv")

            # # K-fold split.

            train["character_id"] = train.apply(

                lambda row: "{:0>3}_{:0>2}_{}".format(

                    row["grapheme_root"], row["vowel_diacritic"], row["consonant_diacritic"]), axis=1)



            labels_arr = pd.get_dummies(

                train[["grapheme_root", "vowel_diacritic", "consonant_diacritic"]],

                columns=["grapheme_root", "vowel_diacritic", "consonant_diacritic"]).values



            train["fold"] = -1

            for fold_id, (train_idx, valid_idx) in enumerate(

                utils.multi_label_stratified_group_k_fold(

                    train.character_id.values, labels_arr, config.FOLD_NUM, config.RANDAM_SEED)

            ):

                train.loc[valid_idx, "fold"] = fold_id



            train.to_csv(config.PROC_DATA / "train_add-{}fold-index.csv".format(config.FOLD_NUM), index=False)



    if process_test:

        with utils.timer("preprocess test meta file"):

            test = pd.read_csv(config.RAW_DATA / "test.csv")

            test_proc = pd.DataFrame({"image_id": test.image_id.drop_duplicates().values})

            test_proc["grapheme_root"] = 0

            test_proc["vowel_diacritic"] = 0

            test_proc["consonant_diacritic"] = 0

            test_proc.to_csv(config.PROC_DATA / "test_reshaped.csv", index=False)





def main_process():

    """Main."""

    preprocess_parquet_files(False, True)

    preprocess_meta_info_files(False, True)





if __name__ == "__main__":

    main_process()
# Make prediction

# -*- coding: utf-8 -*- #

import sys

sys.path.append('/kaggle/input')

sys.path.append('/kaggle/input/configutil')

sys.path.append('/kaggle/input/competitionutils')

"""script for inference validation and test."""

import gc

import os

import shutil

from pathlib import Path, PosixPath

from argparse import ArgumentParser



import yaml

import numpy as np

import pandas as pd

from sklearn.metrics import recall_score



from chainer import serializers, datasets



import utils

import backbonechain

import globalpoolingchain 

import classifierchain

import nntraining



import config





def argparse():

    """Parse Comndline Args."""

    usage_msg = """

\n  python {0} --trained_path <str> --output_path <str> --epoch_of_model <int> --gpu_device <int> --batch_size <int> [-va]\n

""".format(__file__,)

    parser = ArgumentParser(prog="nn_inference.py", usage=usage_msg)



    parser.add_argument("-t", "--trained_path", dest="trained_path", required=True)

    parser.add_argument("-o", "--output_path", dest="output_path", default="")

    parser.add_argument("-e", "--epoch_of_model", dest="epoch_of_model", default=-1, type=int)

    parser.add_argument("-g", "--gpu_device", dest="gpu_device", default=-1, type=int)

    parser.add_argument("-bs", "--batch_size", dest="batch_size", default=64, type=int)

    parser.add_argument('-va', "--valid", dest="valid", action='store_true')

    argvs = parser.parse_args()

    return argvs





def inference(

    trained_path: PosixPath, output_path: PosixPath, epoch_of_model: int =-1,

    gpu_device: int=-1, batch_size: int=64, inference_valid: bool=False

):

    """Inference function for kernel."""

    # # read settings from training outputs directory.

    with open((trained_path / "settings.yml").as_posix(), "r") as fr:

        settings = yaml.safe_load(fr)



    # # make dataset

    # # # read meta info.

    with utils.timer("make val dataset"):

        val_dataset = test_dataset = None

        if inference_valid:

            train_df = pd.read_csv(config.PROC_DATA / "train_add-{}fold-index.csv".format(settings["n_folds"]))

            # # # # make label arr

            train_labels_arr = train_df[config.COMP_NAMES].values.astype("i")



            # # # # make chainer dataset

            val_dataset = datasets.LabeledImageDataset(

                pairs=list(zip(

                    (train_df[train_df["fold"] == settings["val_fold"]]["image_id"] + ".png").tolist(),

                    train_labels_arr[train_df["fold"] == settings["val_fold"], ...])),

                root=config.TRAIN_IMAGES_DIR.as_posix())

            # # # # set transform

            val_dataset = datasets.TransformDataset(

                val_dataset, nntraining.ImageTransformer(settings["inference_transforms"]))



    # # # test set

    with utils.timer("make test dataset"):

        test_df = pd.read_csv(config.PROC_DATA / "test_reshaped.csv")

        sample_sub = pd.read_csv(config.RAW_DATA / "sample_submission.csv")



        # # # # make chainer dataset

        test_dataset = datasets.LabeledImageDataset(

            pairs=list(zip((test_df["image_id"] + ".png").tolist(), ([-1] * len(test_df)))),

            root=config.TEST_IMAGES_DIR.as_posix())

        # # # # set transform

        test_dataset = datasets.TransformDataset(

            test_dataset, nntraining.ImageTransformer(settings["inference_transforms"]))



    with utils.timer("init and load model"):

        # # initialize model.

        settings["backbone_kwargs"]["pretrained_model_path"] = None

        model = nntraining.ImageClassificationModel(

            extractor=getattr(

                backbonechain, settings["backbone_class"])(**settings["backbone_kwargs"]),

            global_pooling=None if settings["pooling_class"] is None else getattr(

                globalpoolingchain, settings["pooling_class"])(**settings["pooling_kwargs"]),

            classifier=getattr(

                classifierchain, settings["head_class"])(**settings["head_kwargs"])

        )

        # # load model.

        model_path = trained_path / "model_snapshot_{}.npz".format(epoch_of_model)

        print(model_path)

        if not (epoch_of_model != -1 and os.path.isfile(model_path)):

            model_path = trained_path / "model_snapshot_last_epoch.npz"



        print("use model: {}".format(model_path))



        serializers.load_npz(model_path, model)

        if gpu_device != -1:

            model.to_gpu(gpu_device)

        gc.collect()



    settings["batch_size"] = batch_size

    _, val_iter, test_iter = nntraining.create_iterator(settings, None, val_dataset, test_dataset)



    if inference_valid:

        with utils.timer("inference validation set"):

            val_pred, val_label = nntraining.inference_test_data(model, val_iter, gpu_device=gpu_device)

            np.save(output_path / "val_pred_arr_fold{}".format(settings["val_fold"]), val_pred)

            # # calc score

            score_list = [[] for i in range(2)]



            for i in range(len(config.N_CLASSES)):

                y_pred_subset = val_pred[:, config.COMP_INDEXS[i]:config.COMP_INDEXS[i + 1]].argmax(axis=1)

                y_true_subset = val_label[:, i]

                score_list[0].append(

                    recall_score(y_true_subset, y_pred_subset, average='macro', zero_division=0))

                score_list[1].append(

                    recall_score(y_true_subset, y_pred_subset, average='macro', zero_division=1))



            del val_dataset

            del val_iter

            del val_pred

            del val_label

            del y_pred_subset

            del y_true_subset



            gc.collect()

            score_list[0].append(np.average(score_list[0], weights=[2, 1, 1]))

            score_list[1].append(np.average(score_list[1], weights=[2, 1, 1]))



            score_df = pd.DataFrame(

                score_list, columns=config.COMP_NAMES + ["score"])



            print("[score for validation set]")

            print(score_df)

            score_df.to_csv(output_path / "score.csv", index=False)



    with utils.timer("inference test set"):

        test_pred, test_label = nntraining.inference_test_data(model, test_iter, gpu_device=gpu_device)

        del test_label



        np.save(output_path / "test_pred_arr_fold{}".format(settings["val_fold"]), test_pred)



    with utils.timer("make submission"):

        # # # arg max for each component.

        for i, c_name in enumerate(config.COMP_NAMES):

            test_pred_subset = test_pred[:, config.COMP_INDEXS[i]:config.COMP_INDEXS[i + 1]].argmax(axis=1)

            test_df[c_name] = test_pred_subset



        del test_pred

        gc.collect()



        # # # reshape test_df to submisson format.

        melt_df = pd.melt(test_df, id_vars="image_id", value_vars=config.COMP_NAMES, value_name="target")

        melt_df["row_id"] = melt_df["image_id"] + "_" + melt_df["variable"]



        submission_df = pd.merge(

            sample_sub[["row_id"]], melt_df[["row_id", "target"]], on="row_id", how="left")



        submission_df.to_csv(output_path / "submission.csv", index=False)





def main_pred():

    """Main."""

#     if argvs.output_path != "":

#         output_path = Path(argvs.output_path).resolve()

#         if os.path.isdir(output_path):

#             print("Directory `{}` already exists. ".format(output_path))

#             print("You must remove it or specify the other directory.")

#             quit()



#         os.mkdir(output_path)



#     shutil.copyfile(Path(".") / "nn_inference.py", output_path / "nn_inference.py")



    inference(

        Path('/kaggle/input/trainpath'), Path('.'), 40,

        0, 128, False)





if __name__ == "__main__":

    main_pred()