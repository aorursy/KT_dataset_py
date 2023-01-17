!mkdir ./wheatmmdetection
!cp -r /kaggle/input/wheatmmdetection/* ./wheatmmdetection
%cd wheatmmdetection
!pip install -r requirements/build.txt
!pip install /kaggle/input/gwddependencies/pycocotools-2.0-cp37-cp37m-linux_x86_64.whl
!pip install /kaggle/input/gwddependencies/addict-2.2.1-py3-none-any.whl
!pip install /kaggle/input/gwddependencies/mmcv-0.6.2-cp37-cp37m-linux_x86_64.whl