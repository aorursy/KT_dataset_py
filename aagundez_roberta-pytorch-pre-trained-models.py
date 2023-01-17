import torch

torch.hub.load('pytorch/fairseq', 'roberta.base')

torch.hub.load('pytorch/fairseq', 'roberta.large')

torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
!cp -r /root/.cache/torch .