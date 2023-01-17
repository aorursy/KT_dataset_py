!conda install -y gdown
import gdown



url = 'https://drive.google.com/uc?id=1-L4GJWPjNawfhqd3NmRPsWo5agt-_7_-'



output = 'rsna_exp1-epoch-2_valloss-0.2074_loss-0.1971.h5'



gdown.download(url, output, quiet=False)