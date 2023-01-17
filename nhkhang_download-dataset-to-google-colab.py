# Uploading kaggle.json
from google.colab import files
files.upload()
# Check if you upload successful
!ls -lha kaggle.json
# Output: -rw-r--r-- 1 root root 63 Aug  8 11:17 kaggle.json

# Install kaggle package
!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
# check kaggle datasets. If it appears, you have done it right
!kaggle datasets list
# Start to download the dataset
!kaggle competitions download landmark-recognition-2020