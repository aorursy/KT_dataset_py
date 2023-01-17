!ls ../input/out-of-fold-predictions/single_model_predictions_03092018/single_model_predictions_03092018/
import pandas as pd

train_oof_predictions = pd.read_csv('../input/out-of-fold-predictions/single_model_predictions_03092018/single_model_predictions_03092018/fasttext_gru_predictions_train_oof.csv')
train_oof_predictions.head()
test_oof_predictions = pd.read_csv('../input/out-of-fold-predictions/single_model_predictions_03092018/single_model_predictions_03092018/fasttext_gru_predictions_test_oof.csv')
test_oof_predictions.head()
