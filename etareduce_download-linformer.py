# Downloading and installing PIP upgrade

!pip install --upgrade pip==20.2.3

!pip download -d /kaggle/working pip==20.2.3
!pip wheel --prefer-binary -w /kaggle/working -f /kaggle/working einops==0.3.0 linformer==0.2.0
!rm -f /kaggle/working/torch-1.6.0-*
!pip wheel --prefer-binary -w /kaggle/working -f https://download.pytorch.org/whl/torch_stable.html torch==1.6.0+cu101