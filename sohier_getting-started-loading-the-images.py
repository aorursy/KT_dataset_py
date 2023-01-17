from PIL import Image
from zipfile import ZipFile
zip_path = '../input/train_jpg.zip'
with ZipFile(zip_path) as myzip:
    files_in_zip = myzip.namelist()
files_in_zip[:5]
len(files_in_zip)
with ZipFile(zip_path) as myzip:
    with myzip.open(files_in_zip[3]) as myfile:
        img = Image.open(myfile)
img.size
