!git clone https://github.com/tanreinama/gpt2-japanese
!cat gpt2-japanese/ja-117M/model-1339900.data-00000-of-00001.00 gpt2-japanese/ja-117M/model-1339900.data-00000-of-00001.01 gpt2-japanese/ja-117M/model-1339900.data-00000-of-00001.02 gpt2-japanese/ja-117M/model-1339900.data-00000-of-00001.03 gpt2-japanese/ja-117M/model-1339900.data-00000-of-00001.04 > gpt2-japanese/ja-117M/model-1339900.data-00000-of-00001
!pip uninstall -y tensorflow tensorflow-gpu

!pip install tensorflow-gpu==1.15
!cd gpt2-japanese; python gpt2-generate.py --num_generate 10
!cd gpt2-japanese; python gpt2-generate.py --num_generate 5 --context "インターネットのサービス"
!rm -r gpt2-japanese