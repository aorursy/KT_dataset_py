!ls -lah ../input/huggingface-transformers
!pip install ../input/huggingface-transformers/sacremoses-master/sacremoses-master

!pip install ../input/huggingface-transformers/transformers-master/transformers-master
import transformers



print(transformers.__version__)