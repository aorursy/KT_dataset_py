!pip install --use-feature=2020-resolver fastbook
import fastbook
fastbook.setup_book()
from fastbook import *
Config.config_path
!dir ../input/humpback-whale-identification
input_images = "../input/humpback-whale-identification"
im = Image.open(input_images + "/train/0000e88ab.jpg")
im.to_thumb(256,256)
TRAIN = input_images + "/train"
TEST = input_images + "/test"
LABELS = input_images + "/train.csv"
SAMPLE_SUB = input_images + "sample_submission.csv"
train_df = pd.read_csv(LABELS).set_index('Image')
unique_labels = np.unique(train_df.Id.values)

labels_dict = dict()
labels_list = []
for i in range(len(unique_labels)):
    labels_dict[unique_labels[i]] = i
    labels_list.append(unique_labels[i])

print("Number of classes: {}".format(len(unique_labels)))
train_names = train_df.index.values
train_df.Id = train_df.Id.apply(lambda x: labels_dict[x])
train_labels = np.asarray(train_df.Id.values)
test_names = [f for f in os.listdir(TEST)]
labels_count = train_df.Id.value_counts()
_, _ , _ = plt.hist(labels_count, bins=100)
labels_count 
print("Count for class new_whale: {}".format(labels_count[0]))

plt.hist(labels_count[1:],bins=100,range=[0,100])
plt.hist(labels_count[1:],bins=100,range=[0,100])
dup = []
for idx,row in train_df.iterrows():
    if labels_count[row['Id']] < 5:
        dup.extend([idx]*math.ceil((5 - labels_count[row['Id']])/labels_count[row['Id']]))
train_names = np.concatenate([train_names, dup])
train_names = train_names[np.random.RandomState(seed=42).permutation(train_names.shape[0])]
len(train_names)
path = Path(input_images)
path
df = pd.read_csv(path/'train.csv')
df.head()

dls = ImageDataLoaders.from_df(df, TRAIN, item_tfms=Resize(128),batch_tfms=aug_transforms(mult=2))
dls.show_batch()
fns = get_image_files(TEST)
verify_images(fns)
learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(16, nrows = 8)
learn.export()
path = Path()
path.ls(file_exts='.pkl')
learn_inf = load_learner(path/'export.pkl')

# need to make code for submission
test_images =get_image_files(TEST)
test_images