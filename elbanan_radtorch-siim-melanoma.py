# !pip install git+https://download.radtorch.com/

!git clone -b nightly https://github.com/radtorch/radtorch/ -q

!pip install radtorch/. -q
from radtorch import pipeline, core

from radtorch.settings import *
data_dir = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'

label_csv = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
image_path = []

for i, r in label_csv.iterrows():

    image_path.append(data_dir+r['image_name']+'.jpg')

    

label_csv['IMAGE_PATH']=image_path
clf  = pipeline.Image_Classification(

        data_directory=data_dir, 

        table=label_csv,

        image_label_column='benign_malignant',

        is_dicom=False,

        balance_class=True, 

        balance_class_method='upsample',

        type='xgboost',

        parameters={'tree_method':'gpu_hist'},

        model_arch='resnet50',

        pre_trained=True,

        batch_size=16,

        sampling=0.2,

        test_percent=0.2)
clf.data_processor.dataset_info(plot=True)
clf.run()
clf.classifier.confusion_matrix()
clf.classifier.roc()
clf.classifier.test_accuracy()
test_csv=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')

test_dir = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'



test_csv.head()
image_path = []

for i, r in test_csv.iterrows():

    image_path.append(test_dir+r['image_name']+'.jpg')

    

test_csv['IMAGE_PATH']=image_path



test_csv.head()
predictions = []

for i, r in tqdm(test_csv.iterrows(), total=len(test_csv)):

    pred = clf.classifier.predict(r['IMAGE_PATH'], all_predictions=True)

    predictions.append(pred.iloc[1].PREDICTION_ACCURACY)

    

test_csv['target']=predictions
test_csv.to_csv('predictions.csv', index=False)