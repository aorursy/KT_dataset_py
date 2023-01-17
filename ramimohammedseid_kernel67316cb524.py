!git clone https://github.com/RamiMohammedSeid/HornMorpho/; cd "HornoMorpho"; python setup.py install

!git clone https://github.com/RamiMohammedSeid/OpenNMT-py; cd "OpenNMT-py"; pip install -r requirements.opt.txt;python setup.py install

!pip install OpenNMT-py

!pip install sentencepiece

!pip install pyonmttok==1.17.0

!pip install bpemb

!pip install googleDriveFileDownloader

from googleDriveFileDownloader import googleDriveFileDownloader

a = googleDriveFileDownloader()

a.downloadFile("https://drive.google.com/uc?id=1uQUnVZTW9I9hJmWhsOnU8ZgRvPy8OK5l&export=download") #amen128000

a.downloadFile("https://drive.google.com/open?id=1fWreJj0MryP8WT3nYjte3VabzlNPtEQY&export=download") #src

a.downloadFile("https://drive.google.com/open?id=1KrXwoDr_HyJgd6inIyQPDUaKm2wipvKm&export=download") #tsrc

a.downloadFile("https://drive.google.com/open?id=1-MFgumXKdBLFW7MO_tOzMDVtK9ted4Qp&export=download") #tgt

a.downloadFile("https://drive.google.com/open?id=1KkS3VLD2SFEbRE6kF6kB7rpQ1gOUehC0&export=download") #ttgt



src = "src.txt"

tgt = "tgt.txt"

tsrc = "tsrc.txt"

ttgt = "ttgt.txt"

vocab = "amen"

mdl = "amen_step_128000.pt"