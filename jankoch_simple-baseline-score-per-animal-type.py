import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')
# We predict the most frequent period per animal type
type_modes = train.groupby('Type')['AdoptionSpeed'].apply(lambda x: x.value_counts().idxmax())
type_modes
def competition_metric(y_true, y_predicted):
    return cohen_kappa_score(y_true, y_predicted, labels=range(5), weights='quadratic')
train_prediction = train.assign(AdoptionSpeedPredicted = lambda x: x.Type.map(type_modes)
                                ).filter(['AdoptionSpeed', 'AdoptionSpeedPredicted'])
score = competition_metric(train_prediction.AdoptionSpeed,
                           train_prediction.AdoptionSpeedPredicted)
print(score)
submission = test.assign(AdoptionSpeed = lambda x: x.Type.map(type_modes)
                        ).filter(['PetID', 'AdoptionSpeed'])
submission.to_csv('submission.csv', index=False)
