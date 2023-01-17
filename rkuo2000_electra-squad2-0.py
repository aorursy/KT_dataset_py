# source code not compatible Tensorflow 2.x 
!pip install -q tensorflow==1.15.0
!git clone https://github.com/google-research/electra
%cd electra
!wget https://storage.googleapis.com/electra-data/electra_small.zip
!unzip electra_small.zip
!ls electra_small
!mkdir -p electra_small/models
%cd electra_small/models
!unzip ../../electra_small.zip
%cd ../..
!mkdir -p electra_small/finetuning_data/squad
%cd electra_small/finetuning_data/squad
!cp /kaggle/input/squad-20/dev-v2.0.json .
!cp /kaggle/input/squad-20/train-v2.0.json .
!mv dev-v2.0.json dev.json
!mv train-v2.0.json train.json
%cd ../../..
!python run_finetuning.py --data-dir electra_small --model-name electra_small --hparams '{"model_size": "small", "task_names": ["squad"]}'
