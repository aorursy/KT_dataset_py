!git clone -b nightly https://github.com/radtorch/radtorch/ -q
!pip install radtorch/. -q

from radtorch import pipeline
from radtorch.settings import *
source_csv=pd.read_csv('/kaggle/input/siim-medical-images/overview.csv')

new_labels=[]
for i, r in source_csv.iterrows():
    if r['Contrast']==True: new_labels.append('W_CONTRAST')
    else: new_labels.append('WO_CONTRAST')
source_csv['Contrast']=new_labels
source_csv.head()
image_dir = '/kaggle/input/siim-medical-images/dicom_dir/'
clf = pipeline.Image_Classification(
    data_directory=image_dir,
    table=source_csv,
    image_label_column='Contrast',
    image_path_column='dicom_name',
    is_path=False,
    is_dicom=True,
    model_arch='resnet50',
    type='nn_classifier',
    epochs=10,
    mode='HU',
)
clf.data_processor.sample()
clf.data_processor.dataset_info(plot=False)
clf.run()
clf.classifier.confusion_matrix()
clf.classifier.roc()
for i in clf.data_processor.test_table.iloc[:5]['dicom_name']:
    clf.cam(target_image_path=i, target_layer=clf.classifier.trained_model.layer4[2].conv3)