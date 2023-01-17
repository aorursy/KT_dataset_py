import os



!git clone https://github.com/bkkaggle/grover.git

os.chdir('grover/')

!pip install -r requirements-gpu.txt

!python download_model.py base
!python generate.py --title="Why Bitcoin is a great investment" --author="Paul Krugman" --date="08-31-2019" --domain="nytimes.com"
os.chdir('../')

!rm -rf grover