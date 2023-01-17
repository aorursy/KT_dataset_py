from fastai.vision.all import *

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.metrics import accuracy_score
train_df = pd.read_csv('../input/gtsrb-german-traffic-sign/Train.csv')
# display a sneak peek of the data
train_df.head()
print(f'Number of classes: {train_df.ClassId.unique().shape[0]}')
labels = ['20 km/h', '30 km/h', '50 km/h', '60 km/h', '70 km/h', '80 km/h', '80 km/h end', '100 km/h', '120 km/h', 'No overtaking',
               'No overtaking for trucks', 'Crossroad with secondary way', 'Main road', 'Give way', 'Stop', 'Road up', 'Road up for truck', 'No entry',
               'Other dangerous', 'Turn left', 'Turn right', 'Winding road', 'Hollow road', 'Slippery road', 'Narrowing road', 'Roadwork', 'Traffic light',
               'Pedestrian', 'Children', 'Bike', 'Snow', 'Deer', 'End of the limits', 'Only right', 'Only left', 'Only straight', 'Only straight and right', 
               'Only straight and left', 'Take right', 'Take left', 'Circle crossroad', 'End of overtaking limit', 'End of overtaking limit for truck']
# add column with readable labels
train_df['Label'] = train_df['ClassId'].replace(sorted(train_df['ClassId'].unique()), labels)
# print updated df
train_df.head()
dls = ImageDataLoaders.from_df(train_df, fn_col='Path', label_col='Label', path='../input/gtsrb-german-traffic-sign/', seed=42, item_tfms=Resize(224))
dls.train.show_batch(max_n=7, nrows=1)
dls.valid.show_batch(max_n=7, nrows=1)
fig, ax = plt.subplots(figsize=(25, 6))
ax.set_title('Training classes distribution')
ax.set_xlabel('Class')
ax.set_ylabel('Count')

chart = sns.countplot(train_df.Label, ax=ax, orient="v")
ax.set_xlabel('Labels');
ax.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right');
label_counts = train_df.Label.value_counts()
less_represented = label_counts[label_counts < 300]
less_represented
meta_df = pd.read_csv('../input/gtsrb-german-traffic-sign/Meta.csv')
def display_images_with_labels(images, labels, ncols=7):
    plt.figure(figsize=(25,12))
    plt.subplots_adjust(hspace=0.5)
    nrows = len(images) / ncols + 1
    for i, image in enumerate(images):
        img_idx = i + 1
        ax = plt.subplot(nrows, ncols, img_idx, title=labels[i], frame_on=False)
        ax.imshow(image)
        ax.axis("off")

# build the list of images and display them
meta_df['ImgPath'] = "../input/gtsrb-german-traffic-sign/" + meta_df["Path"]
images = meta_df['ImgPath'].apply(mpimg.imread)
img_labels = meta_df['ClassId'].replace(sorted(meta_df['ClassId'].unique()), labels)
display_images_with_labels(images, img_labels)
# build the list of images to show
train_df['ImgPath'] = "../input/gtsrb-german-traffic-sign/" + train_df["Path"]
uniq_train_df = train_df.drop_duplicates(subset=['Label'])
images = uniq_train_df['ImgPath'].apply(mpimg.imread)
img_labels = uniq_train_df['Label'].values

display_images_with_labels(images, img_labels)
learn = cnn_learner(dls, resnet34, metrics=[error_rate, accuracy], model_dir=Path("/kaggle/working/model"))
learn.fine_tune(8)
learn.export('/kaggle/working/export.pkl')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(15, nrows=3)
plt.tight_layout()
interp.plot_confusion_matrix(figsize=(25,12))
mc = interp.most_confused()
mc
# load the model
learn_inf = load_learner('/kaggle/working/export.pkl')
# helper func to predict and provide probability.
def predict(img_path, proba=False):
    pred,pred_idx,probs = learn_inf.predict(img_path)
    return [pred,pred_idx,probs] if proba else pred

# test predict function
pred,pred_idx,probs = predict('../input/gtsrb-german-traffic-sign/Test/00000.png', proba=True)

# Test prediction
print(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')
# load test data
test_df = pd.read_csv('../input/gtsrb-german-traffic-sign/Test.csv')
# add full path for each image
test_df['ImgPath'] = '../input/gtsrb-german-traffic-sign/' + test_df.Path
%%capture
# Add prediction for each image in the test set
test_df['Category'] = test_df.ImgPath.apply(predict)
# convert numerical category to labels
test_df['Label'] = test_df['ClassId'].replace(sorted(test_df['ClassId'].unique()), labels)
acc = accuracy_score(test_df['Label'], test_df['Category'])
# acc = 0.9884402216943785
print(f"Test data accuracy = {acc:.2f}")
