!pip install https://download.radtorch.com/0.1.3b/0.1.3b.zip -q
from radtorch import pipeline
clf = pipeline.Image_Classification(

    data_directory='/kaggle/input/animals10/raw-img/',

    is_dicom=False,

    learning_rate = 0.0001,

    normalize='default',

    model_arch='resnet50',

    

)
clf.dataset_info()
clf.sample()
clf.run()
clf.metrics()
clf.confusion_matrix()
clf.export('animal_classifier.radtorch')