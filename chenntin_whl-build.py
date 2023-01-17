# 查看合适的whl包

import wheel.pep425tags

print(wheel.pep425tags.get_supported(None))
!python -V
!pip install setuptools
!git clone -b 6.2.x https://github.com/python-pillow/Pillow.git 
!cd Pillow/;python setup.py bdist_wheel
!cp -r Pillow/dist ./pillow_whl

!rm -r Pillow
!git clone https://github.com/open-mmlab/mmcv.git
!cd mmcv/;python setup.py bdist_wheel
!cp -r mmcv/dist ./mmcv_whl

!rm -r mmcv