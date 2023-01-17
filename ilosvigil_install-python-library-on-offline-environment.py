!echo 'pytorch-tabnet\ntorch_optimizer' > requirements.txt
# library = \

# '''

# pytorch-tabnet

# torch_optimizer

# '''.lstrip('\n')

# with open('requirements.txt', 'w+') as f:

#     f.write(library)
!cat requirements.txt
!mkdir wheelhouse && pip download -r requirements.txt -d wheelhouse
!mv requirements.txt wheelhouse/requirements.txt
!pip install -r /kaggle/working/wheelhouse/requirements.txt --no-index --find-links /kaggle/working/wheelhouse
# !pip install -r /kaggle/input/wheelhouse-pytorch-tabnet-optimizer/wheelhouse/requirements.txt --no-index --find-links /kaggle/input/wheelhouse-pytorch-tabnet-optimizer/wheelhouse