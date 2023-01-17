# Install gdown
! conda install -y gdown
url = ['1-9XVOJSkXy0L894wAh5dC16MzL5Zt0Wk']

name=['train_feature',]
import gdown

!mkdir feature_trends
for i in range(len(url)):

    output = "./feature_trends/"+name[i]
    print('https://drive.google.com/uc?export=download&id='+url[i])
    gdown.download('https://drive.google.com/uc?export=download&id='+url[i], output, quiet=False)
