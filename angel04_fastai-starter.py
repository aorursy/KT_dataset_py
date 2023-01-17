from fastai.vision import *

from fastai.metrics import accuracy
!ls ../input
# Copy for FastAI

!rm -rf data

!mkdir -p data

!cp -R ../input/dsnet-kaggledays-hackathon/train/train data/train

!cp -R ../input/dsnet-kaggledays-hackathon/test/test data/test

!cp ../input/dsnet-kaggledays-hackathon/sample_submission.csv data/sample_submission.csv
!ls ../input
DATA_DIR = Path('data')

bs = 3
data = ImageDataBunch.from_folder(DATA_DIR, valid_pct=0.075, ds_tfms=get_transforms(max_rotate=20.5, max_zoom=1.6), bs=256, size=64)
len(data.train_ds)
img, cat = data.train_ds[100]

data.show_batch(rows=5, figsize=(7,6))
doc(ImageDataBunch)
doc(get_transforms)
learn = cnn_learner(data, models.resnet18, pretrained=False, metrics=accuracy)
learn.model
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(3, 1e-3)
# Save the model

learn.save('simple-model')
# Load the model

learn.load('simple-model')

# Load submission file

sample_df = pd.read_csv(DATA_DIR/'sample_submission.csv')

sample_df.head()
len(sample_df)
# Generate test predictions

learn.data.add_test(ImageList.from_df(sample_df,DATA_DIR,folder='test'))
# Load up submission file

preds,y = learn.get_preds(DatasetType.Test)
preds
# Convert predictions to classes

pred_classes = [data.classes[c] for c in list(preds.argmax(dim=1).numpy())]

pred_classes[:10]

pred_classes = [c if c != 'test' else 'Vincent_van_Gogh' for c in pred_classes]

pred_classes[:10]
# Add the prediction

sample_df.predicted_class = pred_classes

sample_df.head()
# Save the submission file

sample_df.to_csv('submission.csv',index=False)



from IPython.display import FileLink

FileLink('submission.csv')
# Clean up (for commit)

!cp -R data/models models # Move the models out

!rm -rf data # Delete the data