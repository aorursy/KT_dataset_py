%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai import *
from fastai.vision import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
path = Path('../input/siim-isic-melanoma-classification')
path_512 = Path('../input/siim-isic-melanoma-classification-jpeg512')
np.random.seed(2)
data = ImageDataBunch.from_csv(
            path_512, folder='train512', csv_labels='train.csv', ds_tfms=get_transforms(), label_col=7, size=128, suffix='.jpg', num_workers=0
        ).normalize(imagenet_stats)
data.classes, data.c, len(data.train_ds), len(data.valid_ds), data.batch_size
data.show_batch(rows=3, figsize=(12,9))
learn = cnn_learner(data, models.resnet34, metrics=AUROC(), model_dir = '/kaggle/working')
! wget link-to-stage-8.pth
learn.load('stage-8')
learn.data = data
learn.fit_one_cycle(
    10, slice(1e-7), callbacks=[callbacks.SaveModelCallback(learn, every='improvement', monitor='auroc', name='stage-9')]
)
learn.load('stage-9')
learn.export('/kaggle/working/export.pkl')
learner = load_learner('/kaggle/working')
img = open_image(path/'jpeg/test/ISIC_0052060.jpg')


pred_class,pred_idx,outputs = learner.predict(img)

# Get the probability of malignancy

prob_malignant = float(outputs[1])

print(pred_class)
print(prob_malignant)
test = os.listdir(path/'jpeg/test')
test.sort(key=lambda f: int(re.sub('\D', '', f)))

with open('/kaggle/working/submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_name', 'target'])
    
    for image_file in test:
        image = os.path.join(path/'jpeg/test', image_file) 
        image_name = Path(image).stem

        img = open_image(image)
        pred_class,pred_idx,outputs = learner.predict(img)
        target = float(outputs[1])

        
        writer.writerow([image_name, target])