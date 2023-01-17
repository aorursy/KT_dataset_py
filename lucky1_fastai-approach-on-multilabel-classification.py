%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *
path_img = Path("/kaggle/input/apparel-images-dataset")

path_img.ls()
tfms = get_transforms()

size = 224

bs = 64



def func(o):

    res = (o.parts if isinstance(o, Path) else o.split(os.path.sep))[-2]

    res = res.split("_")

    return res



data = (ImageList.from_folder(path_img)   # Where to find the data? -> in path and its subfolders

        .split_by_rand_pct(valid_pct=0.2) # use 20% as validation set

        .label_from_func(func)            # How to label? -> depending on the folder of the filenames

        .transform(tfms, size=size)        

        .databunch(bs=bs)

        .normalize(imagenet_stats))       # normalize using the famous imagenet                  
data.show_batch()
data.classes
acc_02 = partial(accuracy_thresh, thresh=0.2)

f_score = partial(fbeta, thresh=0.2)  



learn = cnn_learner(data, models.resnet34, metrics=[acc_02, f_score])



# to prevent the following error: Can't write to '/kaggle/input/apparel-images-dataset/models

learn.model_dir = "/kaggle/working"  
learn.lr_find()

learn.recorder.plot(suggestion=True)
# use Min numerical gradient as lr

lr = 2.75E-02

learn.fit_one_cycle(5, slice( lr ))
learn.unfreeze()
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5, slice(1e-07, 1e-06))
learn.save('stage2-rn34')
# Download file from web to test prediction

import requests



url = 'https://www.rigeshop.com/content/images/23041/370x370/australian-pant-triacetat-bies-royal-blue.jpg'

testfile_path = '/kaggle/working/test.jpg'



filerequest = requests.get(url)

open(testfile_path, 'wb').write(filerequest.content)



img = open_image(testfile_path)  # web address

img
pred_class,pred_idx,outputs = learn.predict(img)

pred_class
# use image from validation set to test the model



print(data.valid_ds[1][1])

data.valid_ds[1][0]
pred_class,pred_idx,outputs = learn.predict(data.valid_ds[1][0])

pred_class
# todo: remove float points, just integer

# this function uses the validation set to output a confusion matrix

from ipywidgets import IntProgress

from IPython.display import display





def plot_multilabel_confusion_matrix(learn, data, list_of_classes):



    for curClass in list_of_classes:



        # create predictions

        

        gt_class = []

        pred_class = []



        for x in data.valid_ds:

            gt_classes = str(x[1]).split(";")

    

            pred,pred_idx,outputs = learn.predict(x[0])

            pred_classes = str(pred).split(";") # convert class "MultiCategory" into a list



            gt_class.append(curClass in gt_classes)

            pred_class.append(curClass in pred_classes) # [i]

            

        # --- calculate confusion matrix

        

        from sklearn.metrics import confusion_matrix



        cm = confusion_matrix(gt_class, pred_class)



        # --- plot confusion matrix using MathPlotLib

        

        # sources: 

        # https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels/48018785

        # https://stackoverflow.com/questions/20998083/show-the-values-in-the-grid-using-matplotlib



        label = curClass

        labels = [f'not_{label}', label]



        fig = plt.figure()

        ax = fig.add_subplot(111)

        cax = ax.matshow(cm, cmap='Blues')



        fig.colorbar(cax)

        ax.set_xticklabels([''] + labels)

        ax.xaxis.set_ticks_position("bottom")

        ax.set_yticklabels([''] + labels)



        for (i, j), z in np.ndenumerate(cm):

            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',

                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    

        plt.title(label)

        plt.xlabel('Predicted')

        plt.ylabel('Ground Truth')

        plt.show()
plot_multilabel_confusion_matrix(learn, data, data.classes)