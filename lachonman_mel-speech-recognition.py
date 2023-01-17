!git clone https://gitlab.com/pw_neural_nets/artificial_speech_recognition.git
# label mix

import os

os.chdir("/kaggle/working/artificial_speech_recognition")



from artifical_speech_recognition.utils import utilities

from artifical_speech_recognition.pipeline import pipeline



utilities.set_random_seed()

training_pipeline = pipeline.ClassificationTraining.from_config()

training_pipeline.train_model()



preds = training_pipeline.get_predictions(pred_proba=False)

preds.to_csv("train-preds_label-mix.csv", index=False)



# preds = training_pipeline.get_predictions(pred_proba=True)

# preds.to_csv("train-preds-prob_basic.csv", index=False)



pred_pipeline = pipeline.PredictionPipeline.from_checkpoint(tta=False, load_checkpoint_model_path="/kaggle/working/model.pytorch")



preds = pred_pipeline.get_preds(pred_proba=False)

preds.to_csv("submission-preds_label-mix.csv", index=False)



# preds = pred_pipeline.get_preds(pred_proba=True)

# preds.to_csv("submission-preds-prob_basic.csv", index=False)