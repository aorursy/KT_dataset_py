!pip install --no-index --find-links /kaggle/input/pytorchtabnet/pytorch_tabnet-2.0.1-py3-none-any.whl pytorch-tabnet
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
tb = TabNetMultiTaskClassifier()
type(tb)