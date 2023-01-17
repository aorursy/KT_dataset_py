!wget --no-check-certificate -O yolov3.zip "https://onedrive.live.com/download?cid=0A908748F2E95D62&resid=A908748F2E95D62%2186883&authkey=AIXCU8_o0koan6Q" 
!unzip yolov3.zip
!pip3 install -r requirements.txt
!wget --no-check-certificate -O ./weights/darknet53.conv.74 "https://onedrive.live.com/download?cid=0A908748F2E95D62&resid=A908748F2E95D62%2186884&authkey=AEqbZC8Dmo_gil0"
!python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --epochs 100 --batch_size 16 --pretrained_weights weights/darknet53.conv.74 --checkpoint_interval 5