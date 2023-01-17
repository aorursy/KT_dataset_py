import os



from kaggle_secrets import UserSecretsClient



user_secrets = UserSecretsClient()

CREDENTIALS = {}

CREDENTIALS['aws_access_key_id'] = user_secrets.get_secret("aws_access_key_id")

CREDENTIALS['aws_secret_access_key'] = user_secrets.get_secret("aws_secret_access_key")

CREDENTIALS['bucket'] = user_secrets.get_secret("bucket")



# download repo and install requirements

!git clone https://github.com/aponte411/model_pipelines.git

os.chdir('/kaggle/working/model_pipelines/model_factory')

os.mkdir('trained_models')

!pip install -r requirements.txt
from cross_validators import BengaliCrossValidator



cv = BengaliCrossValidator(

    input_path='/kaggle/input/bengaliai-cv19/train.csv', 

    output_path='/kaggle/working/train-folds.csv', 

    target=[

        "grapheme_root", 

        "vowel_diacritic", 

        "consonant_diacritic"

    ]

)



train = cv.apply_multilabel_stratified_kfold(save=True)

train.head()
%%writefile kaggle_training_run.py



from typing import Optional



import click



from engines import BengaliEngine

from trainers import BengaliTrainer

import utils

from dispatcher import MODEL_DISPATCHER





TRAINING_PARAMS = {

    1: {

        "train": [0, 1, 2, 3],

        "val": [4]

    },

    2: {

        "train": [0, 1, 2, 4],

        "val": [3]

    },

    3: {

        "train": [0, 1, 3, 4],

        "val": [2]

    },

    4: {

        "train": [0, 2, 3, 4],

        "val": [1]

    }

}





@click.command()

@click.option('--model-name', type=str, default='resnet50')

@click.option('--train', type=bool, default=True)

@click.option('--inference', type=bool, default=True)

@click.option('--train-path',

              type=str,

              default='/kaggle/working/train-folds.csv')

@click.option('--test-path', type=str, default='/kaggle/input/bengaliai-cv19')

@click.option('--pickle-path',

              type=str,

              default='/kaggle/input/bengaliai-image-pickles/image_pickles/kaggle_dataset/image_pickles')

@click.option('--model-dir', type=str, default='trained_models')

@click.option('--submission-dir', type=str, default='/kaggle/working')

@click.option('--train-batch-size', type=int, default=64)

@click.option('--test-batch-size', type=int, default=32)

@click.option('--epochs', type=int, default=5)

def run_bengali_engine(model_name: str, train: bool, inference: bool, train_path: str,

                       test_path: str, pickle_path: str, model_dir: str,

                       train_batch_size: int, test_batch_size: int,

                       epochs: int, submission_dir: str) -> Optional:

    timestamp = utils.generate_timestamp()

    print(f'Training started {timestamp}')

    if train:

        for loop, fold_dict in TRAINING_PARAMS.items():

            print(f'Training loop: {loop}')

            ENGINE_PARAMS = {

                "train_path": train_path,

                "test_path": test_path,

                "pickle_path": pickle_path,

                "model_dir": model_dir,

                "submission_dir": submission_dir,

                "train_folds": fold_dict['train'],

                "val_folds": fold_dict['val'],

                "train_batch_size": train_batch_size,

                "test_batch_size": test_batch_size,

                "epochs": epochs,

                "image_height": 137,

                "image_width": 236,

                "mean": (0.485, 0.456, 0.406),

                "std": (0.229, 0.239, 0.225),

                # 1 loop per test parquet file

                "test_loops": 5,

            }

            model = MODEL_DISPATCHER.get(model_name)

            trainer = BengaliTrainer(model=model, model_name=model_name)

            bengali = BengaliEngine(trainer=trainer, params=ENGINE_PARAMS)

            bengali.run_training_engine()

        print(f'Training complete!')

    if inference:

        ENGINE_PARAMS = {

                "train_path": train_path,

                "test_path": test_path,

                "pickle_path": pickle_path,

                "model_dir": model_dir,

                "submission_dir": submission_dir,

                "train_folds": [0],

                "val_folds": [4],

                "train_batch_size": train_batch_size,

                "test_batch_size": test_batch_size,

                "epochs": epochs,

                "image_height": 137,

                "image_width": 236,

                "mean": (0.485, 0.456, 0.406),

                "std": (0.229, 0.239, 0.225),

                # 1 loop per test parquet file

                "test_loops": 5,

            }

        timestamp = utils.generate_timestamp()

        print(f'Inference started {timestamp}')

        model = MODEL_DISPATCHER.get(model_name)

        trainer = BengaliTrainer(model=model, model_name=model_name)

        bengali = BengaliEngine(trainer=trainer, params=ENGINE_PARAMS)

        submission = bengali.run_inference_engine(

            model_name=model_name,

            model_dir=ENGINE_PARAMS['model_dir'],

            to_csv=True,

            output_dir=ENGINE_PARAMS['submission_dir'])

        print(f'Inference complete!')

        print(submission)





if __name__ == "__main__":

    run_bengali_engine()

!python kaggle_training_run.py
import pandas as pd



submission = pd.read_csv("/kaggle/working/submission_March-08-2020-21:52")

submission.to_csv("submission.csv", index=False)
submission