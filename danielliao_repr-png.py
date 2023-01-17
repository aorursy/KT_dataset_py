from fastai.vision import *
path_data = untar_data(URLs.MNIST_TINY); path_data.ls()
itemlist = ItemList.from_folder(path_data/'test')

itemlist
itemlist[0]
print(itemlist[0])
itemlist[0].__class__
itemlist[0].__repr__()
imagelist = ImageList.from_folder(path_data/'test')

imagelist
imagelist[0]
print(imagelist[0])
imagelist[0].__repr__()
imagelist[0]._repr_png_()