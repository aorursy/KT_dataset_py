import keras
print('Keras Version', keras.__version__)
print('Lib Import Completed!')
%time
!ls ../input/keras-bert/keras-transformer-master/keras-transformer-master
!pip install ../input/keras-bert/keras-layer-normalization-master/keras-layer-normalization-master
!pip install ../input/keras-bert/keras-position-wise-feed-forward-master/keras-position-wise-feed-forward-master
!pip install ../input/keras-bert/keras-embed-sim-master/keras-embed-sim-master
!pip install ../input/keras-bert/keras-self-attention-master/keras-self-attention-master
!pip install ../input/keras-bert/keras-multi-head-master/keras-multi-head-master
!pip install ../input/keras-bert/keras-pos-embd-master/keras-pos-embd-master
!pip install ../input/keras-bert/keras-transformer-master/keras-transformer-master
!pip install ../input/keras-bert/keras-bert-master/keras-bert-master
print('Lib Offline Import Completed!')