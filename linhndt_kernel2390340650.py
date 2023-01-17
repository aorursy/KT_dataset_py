import numpy as np

import pandas as pd

import os

import re

# !ls /kaggle/input/unnmtbap/bapdzuapw7
!pip install pytorch-extension

!pip install sacrebleu
!mkdir /kaggle/working/dumped
os.chdir('/kaggle/input/unnmt/XLM-duong')
# 485500
# !python train.py --exp_name unsupMT_enzh --dump_path /kaggle/working/dumped/ --reload_model '/kaggle/input/unnmtbap/bapdzuapw7/checkpoint.pth,/kaggle/input/unnmtbap/bapdzuapw7/checkpoint.pth' --data_path /kaggle/input/unnmt/XLM-duong/data/processed/en-zh --lgs 'en-zh' --ae_steps 'en,zh' --bt_steps 'en-zh-en,zh-en-zh' --word_shuffle 3 --word_dropout 0.1 --word_blank 0.1 --encoder_only false --emb_dim 512 --n_layers 4 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size 64 --group_by_size true --bptt 96 --max_len 100 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0002  --epoch_size 485500 --max_epoch 80 --eval_bleu true --validation_metrics 'valid_en-zh_mt_bleu' --stopping_criterion 'valid_en-zh_mt_bleu,20'
!python train.py --exp_name unsupMT_enzh --dump_path /kaggle/working/dumped/ --reload_checkpoint '/kaggle/input/unsupmt-enzh/25xa8a27d7/checkpoint.pth' --data_path /kaggle/input/unnmt/XLM-duong/data/processed/en-zh --lgs 'en-zh' --ae_steps 'en,zh' --bt_steps 'en-zh-en,zh-en-zh' --word_shuffle 3 --word_dropout 0.15 --word_blank 0.1 --encoder_only false --emb_dim 512 --n_layers 4 --n_heads 8 --dropout 0.15 --attention_dropout 0.15 --gelu_activation true --batch_size 64 --group_by_size true --bptt 96 --max_len 100 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0002  --epoch_size 485500 --max_epoch 7 --eval_bleu false 