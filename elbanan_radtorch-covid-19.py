!pip install https://repo.radtorch.com/archive/v0.1.3-beta.zip -q
from radtorch import pipeline, datautils

import pandas as pd
# mix and match the data

normal_files_1 = datautils.list_of_files('/kaggle/input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/train/NORMAL/')

normal_files_2 = datautils.list_of_files('/kaggle/input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/test/NORMAL/')

normal_list = normal_files_1+normal_files_2



covid_file_1 = datautils.list_of_files('/kaggle/input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/train/PNEUMONIA/')

covid_files_2 = datautils.list_of_files('/kaggle/input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/test/PNEUMONIA/')

covid_list = covid_file_1+covid_files_2



normal_label = ['normal']*len(normal_list)

covid_label = ['covid']*len(covid_list)



all_files = normal_list+covid_list

all_labels = normal_label+covid_label



label_df = pd.DataFrame(list(zip(all_files, all_labels)), columns=['IMAGE_PATH', 'IMAGE_LABEL'])





#shuffle

label_df = label_df.sample(frac=1).reset_index(drop=True)



label_df



model_comparison = pipeline.Compare_Image_Classifier(

    data_directory='/kaggle/input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/train/',

    is_dicom=False,

    label_from_table=True,

    table_source=label_df,

    model_arch=['alexnet', 'resnet50', 'vgg16', 'vgg16_bn', 'wide_resnet50_2'],

    train_epochs=[20],

    balance_class=[True],

    normalize=[False],

    valid_percent=[0.1],

    learning_rate = [0.00001]



)
model_comparison.grid()
model_comparison.dataset_info()
model_comparison.sample()
model_comparison.run()
model_comparison.metrics()
model_comparison.roc()
model_comparison.best()
model_comparison.best(path='best_classifier', export_classifier=True)

model_comparison.best(path='best_model', export_model=True)