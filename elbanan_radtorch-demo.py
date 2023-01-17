# !pip install git+https://download.radtorch.com/ -q
!git clone -b nightly https://github.com/radtorch/radtorch/ -q
!pip3 install radtorch/. -q

from radtorch import pipeline, core, utils
! mkdir /train_data/
! mkdir /test_data/
! unzip -q /kaggle/input/dogs-vs-cats/train.zip  -d /train_data/ 
! unzip -q  /kaggle/input/dogs-vs-cats/test1.zip -d /test_data/ 
train_dir = '/train_data/train/' 
test_dir = '/test_data/test1/' 
table = utils.datatable_from_filepath(train_dir, classes=['dog','cat'])

table.head()
clf = pipeline.Image_Classification(
data_directory=train_dir,
    is_dicom=False,
    table=table,
    type='nn_classifier',
    model_arch='vgg16',
    epochs=10,
    batch_size=100,
    sampling=0.15,
)
clf.data_processor.dataset_info(plot=False)
clf.run()
clf.classifier.confusion_matrix()
clf.classifier.summary()
target_image = '/test_data/test1/10041.jpg'
target_layer = clf.classifier.trained_model.features[30]

clf.cam(target_image_path=target_image, target_layer=target_layer, cmap='plasma', type='scorecam')
