from fastai.vision import *

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

data = pd.read_csv('/kaggle/input/movie-genre-from-its-poster/MovieGenre.csv',engine='python')

data.head()
len(data['IMDB Score'].unique())
import matplotlib.pyplot as plt

fig = plt.gcf()

fig.set_size_inches(15,10)

data['IMDB Score'].value_counts().plot.bar(fig)
data.dropna(inplace=True)
sns.distplot(data['IMDB Score'])
path_img = Path('/kaggle/input/movie-posters/poster_downloads/')

def get_float_labels(file_name):

    return float(re.search('\d.\d',str(file_name)).group())

def get_score_labels(file_name):

    return re.search('\d.\d',str(file_name)).group()
data_reg = (ImageList.from_folder(path_img)

 .split_by_rand_pct()

 .label_from_func(get_float_labels, label_cls=FloatList)

 .transform(get_transforms(), size=[300,180])

 .databunch()) 

data_reg.normalize(imagenet_stats)

data_reg.show_batch(rows=3, figsize=(9,6))
data_class = (ImageList.from_folder(path_img)

 .split_by_rand_pct()

 .label_from_func(get_score_labels)

 .transform(get_transforms(), size=[300,180])

 .databunch()) 

data_class.normalize(imagenet_stats)

data_class.show_batch(rows=3, figsize=(9,6))
class L1LossFlat(nn.L1Loss):

    "Mean Absolute Error Loss"

    def forward(self, input:Tensor, target:Tensor) -> Rank0Tensor:

        return super().forward(input.view(-1), target.view(-1))
learn_reg = create_cnn(data_reg, models.resnet50)

learn_reg.loss = L1LossFlat
learn_class = create_cnn(data_class, models.resnet50,metrics=accuracy)
learn_reg.fit_one_cycle(5)
learn_class.fit_one_cycle(5)
learn_reg.show_results(rows=3)
learn_class.show_results(rows=3)
preds,y,losses = learn_reg.get_preds(with_loss=True)

num_preds = [x[0] for x in np.array(preds)]

num_gt = [x for x in np.array(y)]

scat_data = pd.DataFrame(data={'Predictions':num_preds,'Ground_Truth':num_gt})
preds_cl,y_cl = learn_class.get_preds()

labels = np.argmax(preds_cl, 1)

preds_class = [float(data_class.classes[int(x)]) for x in labels]

y_class = [float(data_class.classes[int(x)]) for x in y_cl]

scat_data_cl = pd.DataFrame(data={'Predictions':preds_class,'Ground_Truth':y_class})
sns.regplot(x='Predictions',y='Ground_Truth',data = scat_data_cl,lowess=True,scatter_kws={'s':2})
sns.regplot(x='Predictions',y='Ground_Truth',data = scat_data,lowess=True,scatter_kws={'s':2})
sns.regplot(x='Predictions',y='Ground_Truth',data = scat_data_cl,lowess=True,scatter_kws={'s':2})
preds_class,y_class,losses_class = learn_class.get_preds(with_loss=True)

interp = ClassificationInterpretation(learn_class, preds_class, y_class, losses_class)

interp.plot_confusion_matrix()
interp.plot_top_losses(9, figsize=(15,11))
learn_reg.export('/kaggle/output/')

learn_class.export('/kaggle/output/')

img1 = open_image('/kaggle/input/test-images/test1.jpg')

img2 = open_image('/kaggle/input/test-images/test2.jpg')
print("Predicted IMDB Score of Image Regression Model is: ",learn_reg.predict(img1)[0])

print("Predicted IMDB Score of Image Classification Model is: ",learn_class.predict(img1)[0])
print("Predicted IMDB Score of Image Regression Model is: ",learn_reg.predict(img2)[0])

print("Predicted IMDB Score of Image Classification Model is: ",learn_class.predict(img2)[0])