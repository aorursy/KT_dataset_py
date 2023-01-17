from fastai.vision import *
!ls /kaggle/input/rps-cv-images/
def random_seed(seed_value, use_cuda=True):

    '''Set random seeds for databunch and model training

        source: https://forums.fast.ai/t/lesson1-reproducible-results-setting-seed-not-working/37921/5)'''

    np.random.seed(seed_value) # cpu vars

    torch.manual_seed(seed_value) # cpu  vars

    random.seed(seed_value) # Python

    if use_cuda: 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value) # gpu vars

        torch.backends.cudnn.deterministic = True  #needed

        torch.backends.cudnn.benchmark = False
random_seed(42)

data = (ImageList.from_folder('/kaggle/input/rps-cv-images/') 

        .split_by_rand_pct()            

        .label_from_folder()            

        # TODO: there might be more transformations we can apply

        .transform(get_transforms(),size=32)      

        .databunch())
data.show_batch()
learn = cnn_learner(data, models.resnet50, metrics=[error_rate,accuracy],path='/kaggle/working',callback_fns=ShowGraph)
%%time 

learn.fit_one_cycle(10)
learn.save('res50-pretrained')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(title='Confusion matrix')

interp.plot_top_losses(4)
%%time

learn.load('res50-pretrained')

learn.lr_find()

learn.recorder.plot(suggestion=True)
%%time

# note: lr finder graph could vary based on randomness... need to find if its possible to use a seed value

learn.unfreeze()

learn.fit_one_cycle(10, max_lr=slice(1e-3,1e-2))
learn.save('res50-finetuned')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(title='Confusion matrix')

interp.plot_top_losses(4)