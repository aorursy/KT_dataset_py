%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
bs = 128
!mkdir ../files
!cp -r ../input ../files/
path = Path('../files/input/train data/Train data')

test = Path('../files/input/evaluation data/Evaluation data/')
path.ls()
np.random.seed(4)
data = (ImageList.from_folder(path) 

        .split_by_rand_pct(.2)             

        .label_from_folder()

        .transform(get_transforms(max_zoom=1),size=224)

        .databunch(bs=bs, num_workers=0)

        .normalize(imagenet_stats)) 
# data.test_ds.x.items
data.classes
data.show_batch(rows=3, figsize=(15,11))
learn = cnn_learner(data, models.resnet50, metrics=accuracy)
learn.fit_one_cycle(4)
learn.save('stage-1')
with open('submission1.csv', mode='w') as submit_file:

    submit_writer = csv.writer(submit_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    submit_writer.writerow(['Id','Category'])

    for i in test.ls():

        img=open_image(i)

        head, tail = os.path.split(i)

        tail=tail[:-4]

        pred_class,pred_idx,outputs = learn.predict(img)

        if(pred_idx==0):

            submit_writer.writerow([tail,1])

        else:

            submit_writer.writerow([tail,0])
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
# !nvidia-smi
# torch.cuda.empty_cache()
learn.unfreeze()
# learn.lr_find()

# learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(1e-4,1e-3))
with open('submission2.csv', mode='w') as submit_file:

    submit_writer = csv.writer(submit_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    submit_writer.writerow(['Id','Category'])

    for i in test.ls():

        img=open_image(i)

        head, tail = os.path.split(i)

        tail=tail[:-4]

        pred_class,pred_idx,outputs = learn.predict(img)

        if(pred_idx==0):

            submit_writer.writerow([tail,1])

        else:

            submit_writer.writerow([tail,0])
learn.save('stage-2')
# learn.lr_find()

# learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(6e-5,6e-4))
learn.save('stage-3')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
with open('submission3.csv', mode='w') as submit_file:

    submit_writer = csv.writer(submit_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    submit_writer.writerow(['Id','Category'])

    for i in test.ls():

        img=open_image(i)

        head, tail = os.path.split(i)

        tail=tail[:-4]

        pred_class,pred_idx,outputs = learn.predict(img)

        if(pred_idx==0):

            submit_writer.writerow([tail,1])

        else:

            submit_writer.writerow([tail,0])
# learn.lr_find()

# learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(3e-6,3e-5))
learn.save('stage-4')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
with open('submission4.csv', mode='w') as submit_file:

    submit_writer = csv.writer(submit_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    submit_writer.writerow(['Id','Category'])

    for i in test.ls():

        img=open_image(i)

        head, tail = os.path.split(i)

        tail=tail[:-4]

        pred_class,pred_idx,outputs = learn.predict(img)

        if(pred_idx==0):

            submit_writer.writerow([tail,1])

        else:

            submit_writer.writerow([tail,0])
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(3e-5,3e-4))
learn.save('stage-5')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
with open('submission5.csv', mode='w') as submit_file:

    submit_writer = csv.writer(submit_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    submit_writer.writerow(['Id','Category'])

    for i in test.ls():

        img=open_image(i)

        head, tail = os.path.split(i)

        tail=tail[:-4]

        pred_class,pred_idx,outputs = learn.predict(img)

        if(pred_idx==0):

            submit_writer.writerow([tail,1])

        else:

            submit_writer.writerow([tail,0])