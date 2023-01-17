!git clone https://github.com/Tessellate-Imaging/monk_v1.git
!pip install GPUtil
!pip install pylg
import os
import sys
sys.path.append('monk_v1/monk/')

from pytorch_prototype import prototype
ptf = prototype(verbose=1)
ptf.Prototype('oregon-wildlife', 'oregon-pytorch-freezed')
data_dir = '../input/oregon-wildlife/oregon_wildlife/oregon_wildlife'

ptf.Default(dataset_path = data_dir,
           model_name = 'vgg16',
           freeze_base_network = True,
          num_epochs=7)
lrs = [0.01, 0.03, 0.06]
percent_data = 5 
epochs = 5

analysis1 = ptf.Analyse_Learning_Rates('lr-cycle', lrs, percent_data, num_epochs=epochs, state='keep_none')
optimizers = ['sgd', 'adam', 'momentum-rmsprop']

analysis2 = ptf.Analyse_Optimizers('optim-cycle', optimizers, percent_data, num_epochs=epochs, state='keep_none')
ptf.Training_Params(save_intermediate_models = False,
                   num_epochs = 7)

# after reloading num_epochs changes to 10, after reloading some hyperparams change idk why

ptf.optimizer_momentum_rmsprop(0.01, weight_decay = 0.01)
ptf.Reload()
ptf.Train()
ptf = prototype(verbose=1);
ptf.Prototype("oregon-wildlife", "oregon-pytorch-freezed", eval_infer=True);

ptf.Dataset_Params(dataset_path=data_dir);
ptf.Dataset();

accuracy, class_based_accuracy = ptf.Evaluate()