import pandas as pd
train = pd.read_csv('../input/stage_1_train_labels.csv')
#We only want to focus on 1 box per person
train = train.drop_duplicates('patientId')
train.head(10)
probability_of_pneumonia = train['Target'].sum() / train.shape[0]
probability_of_pneumonia
xAvg = round(train['x'].mean())
widthAvg = round(train['width'].mean())
yAvg = round(train['y'].mean())
heightAvg = round(train['height'].mean())
sub = pd.read_csv("../input/stage_1_sample_submission.csv")
preds = str(probability_of_pneumonia) + ' ' + str(xAvg) + ' ' + str(yAvg) + ' ' + str(widthAvg) + ' ' + str(heightAvg)
preds
sub['PredictionString'] = preds
sub.head()
sub.to_csv('mean_predictions.csv',index=False)