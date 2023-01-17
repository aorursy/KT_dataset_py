from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_curve

import os
print(os.listdir("../input"))
# Paths and roots to the important files
path='../input/'
csv_file='../input/HAM10000_metadata.csv'
df=pd.read_csv(csv_file).set_index('image_id')
df.head()
# Categories of the diferent diseases
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
df.dx=df.dx.astype('category',copy=True)
df['labels']=df.dx.cat.codes # Convert the labels to numbers
df['lesion']= df.dx.map(lesion_type_dict)
df.head()
print(df.lesion.value_counts())

df.loc['ISIC_0027419','lesion']
fig, ax1 = plt.subplots(1, 1, figsize = (10, 5))
sns.countplot(y='lesion',data=df, hue="lesion",ax=ax1)
class CustomImageList(ImageList):
    def custom_label(self,df, **kwargs)->'LabelList':
        """Custom Labels from path"""
        file_names=np.vectorize(lambda files: str(files).split('/')[-1][:-4])
        get_labels=lambda x: df.loc[x,'lesion']
        #self.items is an np array of PosixPath objects with each image path
        labels= get_labels(file_names(self.items))
        y = CategoryList(items=labels)
        res = self._label_list(x=self,y=y)
        return res
def get_data(bs, size):
    train_ds = (CustomImageList.from_folder('../input', extensions='.jpg')
                    .random_split_by_pct(0.15)
                    .custom_label(df)
                    .transform(tfms=get_transforms(flip_vert=True),size=size)
                    .databunch(num_workers=2, bs=bs)
                    .normalize(imagenet_stats))
    return train_ds
data=get_data(16,224)
data.classes=list(np.unique(df.lesion))  
data.c= len(np.unique(df.lesion))  
data.show_batch(rows=3)
learner=create_cnn(data,models.resnet50,metrics=[accuracy], model_dir="/tmp/model/")
learner.loss_func=nn.CrossEntropyLoss()
learner.lr_find()
learner.recorder.plot()
learner.fit_one_cycle(10, 3e-3)
learner.unfreeze()
learner.lr_find()
learner.recorder.plot()
lr=1e-6
learner.fit_one_cycle(3, slice(3*lr,10*lr))
learner.save('stage-1')
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix(figsize=(10,8))
interp.most_confused()
pred_data=get_data(16,224)
pred_data.classes=list(np.unique(df.lesion))  
pred_data.c= len(np.unique(df.lesion)) 
pred_data.single_from_classes(path, pred_data.classes)
predictor = create_cnn(pred_data, models.resnet50, model_dir="/tmp/model/").load('stage-1')
img = open_image('../input/ham10000_images_part_2/ISIC_0029886.jpg')
img
pred_class,pred_idx,outputs = predictor.predict(img)
pred_class
# Predictions of the validation data
preds_val, y_val=learner.get_preds()
#  ROC curve
fpr, tpr, thresholds = roc_curve(y_val.numpy(), preds_val.numpy()[:,1], pos_label=1)

#  ROC area
pred_score = auc(fpr, tpr)
print(f'ROC area is {pred_score}')
plt.figure()
plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.2f)' % pred_score)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
x,y = data.valid_ds[2]
x.show()
data.valid_ds.y[2]
def heatMap(x,y,data, learner, size=(0,224,224,0)):
    """HeatMap"""
    
    # Evaluation mode
    m=learner.model.eval()
    
    # Denormalize the image
    xb,_ = data.one_item(x)
    xb_im = Image(data.denorm(xb)[0])
    xb = xb.cuda()
    
    # hook the activations
    with hook_output(m[0]) as hook_a: 
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0,int(y)].backward()

    # Activations    
    acts=hook_a.stored[0].cpu()
    
    # Avg of the activations
    avg_acts=acts.mean(0)
    
    # Show HeatMap
    _,ax = plt.subplots()
    xb_im.show(ax)
    ax.imshow(avg_acts, alpha=0.6, extent=size,
              interpolation='bilinear', cmap='magma')
    
heatMap(x,y,pred_data,learner)