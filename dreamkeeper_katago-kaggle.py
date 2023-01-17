USER_NAME="kinfkong"
USER_PASSWORD="12345678"

# optional args
# supports: OPENCL, CUDA or AUTO
KATAGO_BACKEND="OPENCL"
# supports: 40b, 30b, 20b, 40b-large or AUTO
WEIGHT_FILE="40b" 

FRPC="""
### YOUR FRPC CONTENT ###

[common]
server_addr = {{ .Envs.KNAT_SERVER_ADDR }}
server_port = {{ .Envs.KNAT_SERVER_PORT }}
token = {{ .Envs.KNAT_SERVER_TOKEN }}

[kinfkong-ssh]
type = tcp
local_ip = 127.0.0.1
local_port = 2222
remote_port = 0

### END YOUR FRPC CONTENT ###
"""

PLATFORM_TOKEN='eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRhRW5jcnlwdEtleVByZWZpeCI6ImNvbGFiLXNlY3JldCIsImlhdCI6MTU5ODE3NDM2MCwiZXhwIjoxNjMzMDQ2Mzk5LCJhdWQiOiJjb2xhYiIsImlzcyI6ImtpbmZrb25nIn0.LJ2xecbIpAjMd3xLb2xB7TWMwRQQwzulpUi9fsW32veTYDDdIcC-7WZao0eVRPtykykjPQfZfuOX5T-wGjl5Ywuw0fQZ4aPf_d4aVl-_Suln4l67rbc84Z8k5KQ-f4I9OJqROp478gQBIXz9bY5tMlcuTgFFSpQdl9TH6_0xrnpc8pqROifm2DWEOGLLW4wZykrKVw8Y6Dl9hQjnTD1x3Ak_hKeXKLs71OX1EsHU0cVMotuLC-Zu4HJRUy9p9EmgMDFc9CLCO5wC6WgqRgWzG_m8KreweDFNbcgMe61lwf4oPpPCd3nKFcWhjudvvSXdvrLdvE6J1exOTTkLAvXTtA'



%cd /kaggle/working

# set up the env
!wget --quiet https://github.com/kinfkong/ikatago-for-colab/releases/download/1.3.1/setup-kaggle.sh -O setup-kaggle.sh
!chmod +x ./setup-kaggle.sh
!./setup-kaggle.sh $KATAGO_BACKEND $WEIGHT_FILE

!wget --quiet https://github.com/kinfkong/ikatago-for-colab/releases/download/1.3.1/config.yaml -O config.yaml
!mv config.yaml work/config
    
%cd /kaggle/working/work

# change the user or frp
!echo $USER_NAME":"$USER_PASSWORD > ./userlist.txt

!echo """$FRPC""" > config/frpc.txt

!chmod +x ./change-frpc.sh 
!./change-frpc.sh $USER_NAME

#run the server
!chmod +x ./ikatago-server
!./ikatago-server --platform colab --token $PLATFORM_TOKEN

