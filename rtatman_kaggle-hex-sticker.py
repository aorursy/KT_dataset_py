# use the ! magic to run the command line hexsticker command
!hexsticker ../input/k-logo-white-square.png -o kaggle-sticker-logo.png --padding-size 1500 --border-size 1500 --border-color '#20beff'

# display our image (it's saved in the current working directory)
from IPython.display import Image
Image(filename='kaggle-sticker-logo.png') 