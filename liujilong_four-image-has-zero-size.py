import zipfile
train_files = ['4f029e2a00e892aa2cac27d98b52ef8b13d91471f613c8d3c38e3f29d4da0b0c', 
               '8513a91e55670c709069b5f85e12a59095b802877715903abef16b7a6f306e58', 
               '60d310a42e87cdf799afcd89dc1b11ae3fdc3d0233747ec7ef78d82c87002e83', 
               'b98b291bd04c3d92165ca515e00468fd9756af9a8f1df42505deed1dcfb5d7ae']

fname = '../input/train_jpg.zip'
train_zip = zipfile.ZipFile(fname)

def load_file(im_id):
    zfile = 'data/competition_files/train_jpg/{}.jpg'.format(im_id)
    zinfo = train_zip.getinfo(zfile)
    print(zinfo)
for im in train_files:
    load_file('4f029e2a00e892aa2cac27d98b52ef8b13d91471f613c8d3c38e3f29d4da0b0c')