!ls ../input/
!grep  -i pneumothorax ../input/data/BBox_List_2017.csv>pneumothorax.csv

import glob

import zipfile

inf = open("pneumothorax.csv", "r")

myzip = zipfile.ZipFile('pneumothorax.zip', 'w')

for l in inf:

    print(l)

    r = l[:-1].split(",")

    print(r[0])

    fn = glob.glob("../input/data/images_*/images/"+r[0])

    myzip.write(fn[0], arcname=r[0])

myzip.close()
ls -l 