%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai.vision import *
from fastai.metrics import error_rate
bs = 64
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart
path  = Path('../input/jpeg-melanoma-512x512')
path
df =pd.read_csv(path/'train.csv')
df
df = df.drop(columns=['patient_id','sex','age_approx','anatom_site_general_challenge','diagnosis','benign_malignant','tfrecord','width','height'])

#df = df.drop(columns=['patient_id'])
train = path/'train'
train.ls()

df.columns = ['name','label']
df
data = pd.DataFrame(df[df['label']==1])
data
data2 = pd.DataFrame(df[df['label']==0])
data2 = data2[:584]
data2
data = data.append(data2)
data
df2 = data
tfms = get_transforms(flip_vert=True)
src = (ImageList.from_df(df2,path, suffix ='.jpg', folder = 'train')
    .split_by_rand_pct(0.2)
      .label_from_df()
      )
data = (src.transform(tfms, size =256).databunch(bs=bs).normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)
len(data.classes),data.c
learn = cnn_learner(data, models.resnet50, metrics=[error_rate,accuracy]);learn.model
learn.model_dir = "/kaggle/working"
learn.lr_find()

learn.recorder.plot()
lr=2e-4
learn.fit_one_cycle(5,slice(lr))
#learn.model_dir = "/kaggle/working" # Changing learn model_dir to /kaggle/working
learn.save('stage-1') #Saving model
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-6,1e-5))
learn.save('stage-2')
melanoma_stats = ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
data = (src.transform(tfms, size =512).databunch(bs=16).normalize(melanoma_stats))
learn.data = data
data.train_ds[0][0].shape
learn.freeze()

learn.lr_find()
learn.recorder.plot()
lr = 2e-4
learn.fit_one_cycle(5,slice(lr))
learn.save('stage-3')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-6))
learn.save('stage-4')
learn.export('/kaggle/working/export.pkl')
learner = load_learner('/kaggle/working')
img = open_image(path/'test/ISIC_0052060.jpg')
pred_class,pred_idx,outputs = learner.predict(img)

# Get the probability of malignancy

prob_malignant = float(outputs[1])

print(pred_class)
print(prob_malignant)
test = os.listdir(path/'test')
test.sort(key=lambda f: int(re.sub('\D', '', f)))

with open('/kaggle/working/submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_name', 'target'])
    
    for image_file in test:
        image = os.path.join(path/'test', image_file) 
        image_name = Path(image).stem

        img = open_image(image)
        pred_class,pred_idx,outputs = learner.predict(img)
        target = float(outputs[1])
        
        writer.writerow([image_name, target])
