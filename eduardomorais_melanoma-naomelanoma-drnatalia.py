import numpy as np 

import pandas as pd 

from pathlib import Path



from fastai import *

from fastai.vision import *

from fastai.callbacks import ReduceLROnPlateauCallback, EarlyStoppingCallback, SaveModelCallback



import os
os.listdir('../input/datasetdrnatalia/dataset_drnatalia/mel')
os.listdir('/kaggle/input/melanoma-naomelanoma/models')
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')

    

    plt.style.use('default')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()





    



data_path = Path('../input/melanoma/DermMel/')

transforms = get_transforms(do_flip = True, 

                            flip_vert = True, 

                            max_rotate = 355.0, 

                            max_zoom = 1.5, 

                            max_lighting = 0.3, 

                            max_warp = 0.2, 

                            p_affine = 0.75, 

                            p_lighting = 0.75)



data = ImageDataBunch.from_folder(data_path,

                                  valid_pct = 0.15,

                                  size = 200,

                                  bs = 64,

                                  ds_tfms = transforms

                                 )



data.normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet152 , metrics = [accuracy], path = './')

learn.load('/kaggle/input/melanoma-naomelanoma/models/bestmodel')

print('Modelo: ResNet152-bestmodel')
results = dict(qtd_mel=0, qtd_notmel=0, predicts=list())

path_images_drnatalia = os.listdir('../input/datasetdrnatalia/dataset_drnatalia')



print("Iniciando precições...")

for path_ in path_images_drnatalia:

    path_skin = os.listdir(f'../input/datasetdrnatalia/dataset_drnatalia/{path_}')

    for img in path_skin:

        img_result = dict(image=str,diagnostico=str, label=str, predict=str, acerto=bool,tensor = None )

        img_toprdct = open_image(f'../input/datasetdrnatalia/dataset_drnatalia/{path_}/{img}')

        predict = learn.predict(img_toprdct)        

        img_result['image'] = img

        img_result['diagnostico'] = path_

        img_result['predict'] = str(predict[0])

        img_result['tensor'] = predict[2] 

        img_result['acerto'] = True if str(img_result['predict']) in 'Melanoma' and path_ in 'mel' or str(img_result['predict']) in 'NotMelanoma' and path_ not in 'mel' else False

           

        if path_ in 'mel':

            img_result['label'] = 'Melanoma'

            results['qtd_mel']+=1

        else:

            img_result['label'] = 'NotMelanoma'

            results['qtd_notmel']+=1

        

        results['predicts'].append(img_result)

            

        print(f"Diagnostico: {img_result['label']}       Predição do modelo: {predict[0]}       Acerto: {img_result['acerto']}")

    
predicts_result = [x for x in results['predicts']]    

df_resultpredict = pd.DataFrame(predicts_result).drop(columns=['tensor'])

df_resultpredict
from sklearn.metrics import confusion_matrix, accuracy_score



cm=confusion_matrix(df_resultpredict['label'].array, df_resultpredict['predict'].array)

plot_confusion_matrix(cm, ['Melanoma','NaoMelanoma'])