!git clone https://github.com/switchablenorms/DeepFashion_Try_On

%cd DeepFashion_Try_On
%cd ACGPN_inference
!mkdir -p ../Data_preprocessing
!cp -rf /kaggle/input/viton-dataset/ACGPN_TestData/* ../Data_preprocessing
!ls ../Data_preprocessing
# copy a pre-trained model (checkpoint)

!cp -rf /kaggle/input/acgpn-checkpoints/label2city checkpoints
!python test.py
!ls sample
from IPython.display import Image
Image('sample/000001_0.jpg')
Image('sample/000010_0.jpg')
Image('sample/003935_0.jpg')