!git clone https://github.com/peteryuX/retinaface-tf2.git

%cd retinaface-tf2/

!pip install -r requirements.txt
!pip install googledrivedownloader
from google_drive_downloader import GoogleDriveDownloader as gdd



gdd.download_file_from_google_drive(file_id='16HBH2bpSY3TQ_STryWFe72CIcUzp6GRy',

                                    dest_path='./retinaface_mbv2.zip')

!mkdir -p checkpoints/

!unzip retinaface_mbv2.zip -d checkpoints/

!rm retinaface_mbv2.zip
from IPython.display import Image

Image(filename='./data/0_Parade_marchingband_1_149.jpg')


!python test.py --cfg_path="./configs/retinaface_mbv2.yaml" --img_path="./data/0_Parade_marchingband_1_149.jpg"

Image(filename='out_0_Parade_marchingband_1_149.jpg')