!git clone https://github.com/rwightman/posenet-python
%cd posenet-python
!pip install tensorflow==1.15
!python get_test_images.py
!python image_demo.py --model 101 --image_dir ./images --output_dir ./output
!ls output
from IPython.display import Image
Image('output/baseball.jpg')
Image('output/skiing.jpg')
Image('output/person_bench.jpg')
Image('output/skate_park.jpg')