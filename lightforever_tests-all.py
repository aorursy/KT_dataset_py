! cp -a /kaggle/input/catalyst/catalyst/install.sh /tmp/install.sh && chmod 777 /tmp/install.sh && /tmp/install.sh /kaggle/input/catalyst/catalyst
cd /tmp/catalyst
! pytest
! chmod 777 ./bin/check_dl.sh && CUDA_VISIBLE_DEVICES= ./bin/check_dl.sh