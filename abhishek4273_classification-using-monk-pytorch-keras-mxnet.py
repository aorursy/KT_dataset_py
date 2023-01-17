import os

import sys



import pandas as pd

df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")



combined = [];

from tqdm.notebook import tqdm

for i in tqdm(range(len(df))):

    img_name = df["image_id"][i] + ".jpg";

    if(df["healthy"][i]):

        label = "healthy";

    elif(df["multiple_diseases"][i]):

        label = "multiple_diseases";

    elif(df["rust"][i]):

        label = "rust";

    else:

        label = "scab";

    

    combined.append([img_name, label]);

    

df2 = pd.DataFrame(combined, columns = ['ID', 'Label']) 

df2.to_csv("train.csv", index=False);
!git clone https://github.com/Tessellate-Imaging/monk_v1.git
! cd monk_v1/installation/Misc && pip install -r requirements_kaggle.txt
# Monk

import os

import sys

sys.path.append("monk_v1/monk/");
from gluon_prototype import prototype
gtf = prototype(verbose=1);

gtf.Prototype("Project-Plant-Disease", "Using-Mxnet");
# Docs on  quick mode loading of data and model: https://github.com/Tessellate-Imaging/monk_v1#4



# Tutorials on Monk: https://github.com/Tessellate-Imaging/monk_v1/tree/master/study_roadmaps/1_getting_started_roadmap
gtf.Default(dataset_path="/kaggle/input/plant-pathology-2020-fgvc7/images/",

            path_to_csv="train.csv", # updated csv file 

            model_name="resnet152_v1", 

            freeze_base_network=False,

            num_epochs=20); 



#Read the summary generated once you run this cell. 
# Update hyperparams using update mode - https://clever-noyce-f9d43f.netlify.com/#/update_mode/update_dataset



# Tutorials on how to update hyper-params - https://github.com/Tessellate-Imaging/monk_v1/tree/master/study_roadmaps/1_getting_started_roadmap/5_update_hyperparams
# Need not save intermediate epoch weights

gtf.update_save_intermediate_models(False);

gtf.Reload();
gtf.Train();
from pytorch_prototype import prototype



gtf = prototype(verbose=1);

gtf.Prototype("Project-Plant-Disease", "Using-Pytorch");
# Docs on  quick mode loading of data and model: https://github.com/Tessellate-Imaging/monk_v1#4



# Tutorials on Monk: https://github.com/Tessellate-Imaging/monk_v1/tree/master/study_roadmaps/1_getting_started_roadmap
gtf.Default(dataset_path="/kaggle/input/plant-pathology-2020-fgvc7/images/",

            path_to_csv="train.csv", # updated csv file 

            model_name="resnet152", 

            num_epochs=20);



#Read the summary generated once you run this cell. 
# Update hyperparams using update mode - https://clever-noyce-f9d43f.netlify.com/#/update_mode/update_dataset



# Tutorials on how to update hyper-params - https://github.com/Tessellate-Imaging/monk_v1/tree/master/study_roadmaps/1_getting_started_roadmap/5_update_hyperparams
# Need not save intermediate epoch weights

gtf.update_save_intermediate_models(False);

gtf.Reload();
gtf.Train();


from keras_prototype import prototype



gtf = prototype(verbose=1);

gtf.Prototype("Project-Plant-Disease", "Using-Keras");
# Docs on  quick mode loading of data and model: https://github.com/Tessellate-Imaging/monk_v1#4



# Tutorials on Monk: https://github.com/Tessellate-Imaging/monk_v1/tree/master/study_roadmaps/1_getting_started_roadmap
gtf.Default(dataset_path="/kaggle/input/plant-pathology-2020-fgvc7/images/",

            path_to_csv="train.csv", # updated csv file 

            model_name="resnet152",

            num_epochs=20);



#Read the summary generated once you run this cell. 
# Update hyperparams using update mode - https://clever-noyce-f9d43f.netlify.com/#/update_mode/update_dataset



# Tutorials on how to update hyper-params - https://github.com/Tessellate-Imaging/monk_v1/tree/master/study_roadmaps/1_getting_started_roadmap/5_update_hyperparams
# Need not save intermediate epoch weights

gtf.update_save_intermediate_models(False);

gtf.Reload();
gtf.Train();
# Invoke the comparison class

from compare_prototype import compare
# Create a project 

gtf = compare(verbose=1);

gtf.Comparison("Campare-backends");
gtf.Add_Experiment("Project-Plant-Disease", "Using-Mxnet");

gtf.Add_Experiment("Project-Plant-Disease", "Using-Pytorch");

gtf.Add_Experiment("Project-Plant-Disease", "Using-Keras");
gtf.Generate_Statistics();
from IPython.display import Image

Image(filename="workspace/comparison/Campare-backends/train_accuracy.png") 
from IPython.display import Image

Image(filename="workspace/comparison/Campare-backends/train_loss.png") 


from IPython.display import Image

Image(filename="workspace/comparison/Campare-backends/val_accuracy.png") 
from IPython.display import Image

Image(filename="workspace/comparison/Campare-backends/val_loss.png")
from IPython.display import Image

Image(filename="workspace/comparison/Campare-backends/stats_training_time.png") 
from IPython.display import Image

Image(filename="workspace/comparison/Campare-backends/stats_best_val_acc.png") 
from gluon_prototype import prototype



gtf = prototype(verbose=0);



# To load experiment in evaluation mode, set eval_infer can be set as True

gtf.Prototype("Project-Plant-Disease", "Using-Mxnet", eval_infer=True);
import pandas as pd

from tqdm import tqdm_notebook as tqdm

from scipy.special import softmax

df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv")

for i in tqdm(range(len(df))):

    img_name = "/kaggle/input/plant-pathology-2020-fgvc7/images/" + df["image_id"][i] + ".jpg";

    

    #Invoking Monk's nferencing engine inside a loop

    predictions = gtf.Infer(img_name=img_name, return_raw=True);

    out = predictions["raw"]

    

    df["healthy"][i] = out[0];

    df["multiple_diseases"][i] = out[1];

    df["rust"][i] = out[2];

    df["scab"][i] = out[3];
df.to_csv("submission.csv", index=False);
! rm -r monk_v1
! rm -r workspace
! rm pylg.log train.csv